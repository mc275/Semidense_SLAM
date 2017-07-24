


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <mutex>

namespace ORB_SLAM2
{

    // 构造函数，成员变量初始化。
    LocalMapping::LocalMapping( Map *pMap, const float bMonocular):
        mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
        mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
    {

    }
    

    // 设置进程间的对象指针，用于数据交互。
    void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
    {
        mpLoopCloser = pLoopCloser;
    }
    void LocalMapping::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }


    // Local Mapping线程入口函数。
    void LocalMapping::Run()
    {

        mbFinished = false;

        while(1)
        {

            // 告知Tracking线程 Local Mapping线程处于忙碌状态。
            // Local Mapping处理的线程都是Tracking发到mlNewKeyFrames。
            // 没有处理完以前不发送，设置mbAcceptKeyFrames变量表示状态。
            SetAcceptKeyFrames(false);

            // 等待处理的关键帧队列mlNewKeyFrames不能为空。
            if(CheckNewKeyFrames())
            {
                // 步骤1 从队列中取出一帧，计算特征点的BoW向量，将关键帧插入到地图。
                ProcessNewKeyFrame();

                // 步骤2 剔除ProcessNewKeyFrame()中引入的不合格的MapPoints。
                MapPointCulling();

                // 步骤3 运动过程中，与相邻的关键帧三角化恢复出一些MapPoints。
                CreateNewMapPoints();

                // 当前帧是Tracking插入队列中最后一帧关键帧(ProcessNewKeyFrame函数会pop队列)。
                if( !CheckNewKeyFrames() ) 
                {
                    // 检查当前帧与相邻关键帧重复的MapPoints，并融合。
                    SearchInNeighbors();
                }

                mbAbortBA = false;
                
                // 当前帧是队列中的最后一帧，且闭环检测没有发送中断申请。
                if( !CheckNewKeyFrames() && !stopRequested())
                {
                    // 步骤4 局部BA
                    if(mpMap->KeyFramesInMap()>2)
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

                    // 步骤5 剔除冗余关键帧。某关键帧90%的点云被其他关键帧观测，剔除。
                    // Tracking 线程通过InsertKeyFrame函数添加的条件宽松，KF会比较多，便于跟踪。
                    // 在这个删除冗余关键帧。
                    KeyFrameCulling();
                }

                // 将当前关键帧加入闭环检测队列。
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

            }   // 当前关键帧队列不为空。

            // 停止Local Mapping
            else if(Stop())
            {
                // 收到停止请求，但是进程没有完成。
                while( isStopped() && !CheckFinish() )
                {
                    // 延时3000us
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
                // 完成Local Mapping。
                if(CheckFinish())
                    break;
            }

            // 是否进行重置。
            ResetIfRequested();

            // Tracking查看跟踪进程空闲。
            SetAcceptKeyFrames(true);

            // 进程完成。
            if(CheckFinish())
                break;

            // 延时3000us
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }   // while(1) 

        // Local Mapping线程完成。    
        SetFinish();

    }



    // 插入关键帧。
    // Tracking线程在CreateKeyFrame()中调用。
    // 插入关键帧到mlNewKeyFrames。
    // 仅仅是将关键帧插入到队列，之后从队列中pop。
    void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewKFs);

        // 关键帧插入到列表中。
        mlNewKeyFrames.push_back(pKF);
        mbAbortBA = true;
    }



    // 查看队列mlNewKeyFrames中是否还有等待插入的关键帧。
    // return 若果还有，true。
    bool LocalMapping::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        return (!mlNewKeyFrames.empty());
    }



    // 处理队列中的关键帧。
    // 1.计算BoW，加速匹配和三角化新的MapPoints.
    // 2.新匹配的MapPoints关联当前帧。
    // 3.插入关键帧，更新Covisibility图和Essential图。
    void LocalMapping::ProcessNewKeyFrame()
    {

        // 步骤1 从队列中取一帧关键帧。
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            mpCurrentKeyFrame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();
        }

        // 步骤2 计算当前关键帧的BoW向量。
        mpCurrentKeyFrame->ComputeBoW();

        // 步骤3 跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定。
        // TrackLocalMap()只对局部地图中的MapPoints与当前帧进行了匹配，但没有关联。
        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    // 为当前帧在tracking过程跟踪到的MapPoint添加属性。
                    if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        // 添加观测。
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                        // 获得该点的平均观测方向。
                        pMP->UpdateNormalAndDepth();
                        // 加入关键帧后，更新3D点的最佳描述子。
                        pMP->ComputeDistinctiveDescriptors();
                    }

                    // 只发生在双目或者RGBD跟踪过程中插入的MapPoints
                    else
                    {
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }  
                }
            }
        }

        // 步骤4 更新关键帧间的连接关系，Convisible图和Essential图(tree)
        mpCurrentKeyFrame->UpdateConnections();
        
        // 步骤5 将关键帧插入到地图中。
        mpMap->AddKeyFrame(mpCurrentKeyFrame);

    }

    

    // 剔除ProcessingNewKeyFrame和CreateNewMapPoints函数引入质量不好的MapPoints
    void LocalMapping::MapPointCulling()
    {
        // 检查最近添加的MapPoints。
        // 在CreateNewMapPoints中有新添加点云。
        // 单目在ProcessingNewKeyFrame中没有添加点云，双目RGBD会引入。
        list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();          
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;
        if(mbMonocular)
            nThObs = 2;
        else
            nThObs = 3;
        const int cnThObs = nThObs;

        // 遍历所有需要检查的MapPoints。
        while(lit != mlpRecentAddedMapPoints.end())
        {
            MapPoint *pMP = *lit;
            // 步骤1 已经是坏点的直接剔除。
            if(pMP->isBad())
            {
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            
            // 步骤2 跟踪到该MapPoint的Frame数小于预测可跟踪的帧数的0.25，剔除。
            // IncreaseFound / IncreaseVisible < 25%，不一定是关键帧。
            else if(pMP->GetFoundRatio()<0.25f)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }

            // 步骤3 该MapPoint建立开始>=2帧,但是能观测点的关键帧数不超过cnThobs，剔除，单目阈值2。
            else if( ((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }

            // 步骤4 从建立开始，超过3帧，剔除。
            else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
                lit = mlpRecentAddedMapPoints.erase(lit);
            else 
                lit++;
        }

    }



    // 相机运动过程中和共视程度比较高的关键帧通过三角化会出一些MapPoints。
    void LocalMapping::CreateNewMapPoints()
    {
        // 共视帧阈值设置。
        int nn=10;
        if(mbMonocular)
            nn = 20;

        // 步骤1 在Covisible图中找到当前关键帧共视程度最高的nn帧相邻帧。
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        ORBmatcher matcher(0.6, false);

        // 当前帧相对初始世界坐标的位姿。
        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3,4,CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0,3));
        tcw1.copyTo(Tcw1.col(3));

        // 得到当前关键帧在世界坐标系中的坐标。
        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // 搜索极线匹配约束，三角化。
        // 步骤2 遍历所有相邻关键帧vpNeightKFs。
        for(size_t i=0; i<vpNeighKFs.size(); i++)
        {
           if(i>0 && CheckNewKeyFrames()) 
               return;

           KeyFrame *pKF2 = vpNeighKFs[i];

           // 临接关键帧在世界坐标系中的坐标。
           cv::Mat Ow2 = pKF2->GetCameraCenter();
           // 基线向量，两个关键帧之间的位移。
           cv::Mat  vBaseline = Ow2-Ow1;
           // 基线长度。
           const float baseline = cv::norm(vBaseline);

           // 步骤3 判断基线是否足够长。
           // 立体相机。
           if(!mbMonocular)
           {
               if(baseline<pKF2->mb)
                   continue;
           }
           // 单目。
           else
           {
               // 邻接关键帧的场景深度中值。
               const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
               // 基线与场景深度的比值。
               const float ratioBaselineDepth = baseline/medianDepthKF2;
               // 如果比值特别小，不考虑当前邻接的关键帧，不生成3D点。
               if(ratioBaselineDepth<0.01)
                   continue;
           }
           
           // 步骤4 根据两关键帧之间的位姿计算他们的基本矩阵。 
           cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

           // 步骤5 通过极线约束限制匹配的搜索范围，进行特征点匹配。
           vector<pair<size_t, size_t> > vMatchedIndices;
           matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

           cv::Mat Rcw2 = pKF2->GetRotation();
           cv::Mat Rwc2 = Rcw2.t();
           cv::Mat tcw2 = pKF2->GetTranslation();
           cv::Mat Tcw2(3, 4, CV_32F);
           Rcw2.copyTo(Tcw2.colRange(0,3));
           tcw2.copyTo(Tcw2.col(3));

          const float &fx2 = pKF2->fx; 
          const float &fy2 = pKF2->fy;
          const float &cx2 = pKF2->cx;
          const float &cy2 = pKF2->cy;
          const float &invfx2 = pKF2->invfx;
          const float &invfy2 = pKF2->invfy;

          // 步骤6 对每对匹配通过三角化生成3D点。
          const int nmatches = vMatchedIndices.size();
          for(int ikp = 0; ikp<nmatches; ikp++)
          {
              // 步骤6.1 取出匹配的特征点。

              // 当前匹配对在当前关键帧中的索引。
              const int &idx1 = vMatchedIndices[ikp].first;

              // 当前匹配对在邻接关键帧中的索引。
              const int &idx2 = vMatchedIndices[ikp].second;

              // 当前匹配对在当前关键帧中的特征点。
              const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
              // mvuRight存放着双目的深度值，如果不是双目，为-1。
              const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
              bool bStereo1 = kp1_ur>=0;

              // 当前匹配对在邻接关键帧中的特征点。
              const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
              // mvuRight存放双目深度，不是双目时，为-1。
              const float kp2_ur = pKF2->mvuRight[idx2];
              bool bStereo2 = kp2_ur>=0;

              // 步骤6.2 利用匹配点反投影得到视角差。
              // 特征点反投影。
              cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
              cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

              // 由相机坐标系转到世界坐标系，得到视角差余弦值。
              cv::Mat ray1 = Rwc1*xn1;
              cv::Mat ray2 = Rwc2*xn2;
              const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

              // 加1是为了让cosParallaxStereo随便初始化为一个很大的值。
              float cosParallaxStereo = cosParallaxRays+1;
              float cosParallaxStereo1 = cosParallaxStereo;
              float cosParallaxStereo2 = cosParallaxStereo;

              // 步骤6.3 对于双目，利用双目得到视差角。
              if(bStereo1)  // 双目，有深度。
                  cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2, mpCurrentKeyFrame->mvDepth[idx1]));
              else if(bStereo2) // 双目，有深度。
                  cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2, pKF2->mvDepth[idx2]));

              // 得到双目观测的视差角。
              cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

              // 步骤6.4 三角化恢复3D点。
              cv::Mat x3D;
              
              // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表示视角差正常。
              // cosParallaxRays < cosParallaxStereo表示视角小。

              // 视角正常但是小视角。
              if( cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998) && cosParallaxRays < cosParallaxStereo )
              {
                  // 线性三角化方法，见Initialize.cpp的Triangulate函数。
                  cv::Mat A(4, 4, CV_32F);
                  A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                  A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                  A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                  A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                  cv::Mat w, u, vt;
                  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                  x3D = vt.row(3).t();

                  if(x3D.at<float>(3) == 0)
                      continue;

                  // 欧式坐标。
                  x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
              }

              else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
              {
                  x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
              }

              else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
              {
                  x3D = pKF2->UnprojectStereo(idx2);
              }

              // 视差很小。
              else
                  continue;

              cv::Mat x3Dt = x3D.t();

              // 步骤6.5 检测生成的3D点是否在相机前方。
              float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
              if(z1<=0)
                  continue;

              float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
              if(z2<=0)
                  continue;

              // 步骤6.6 计算当前3D点在当前关键帧下的重投影误差。
              const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
              const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
              const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
              const float invz1 = 1.0/z1;

              // 不是双目。
              if(!bStereo1)
              {
                  float u1 = fx1*x1*invz1+cx1;
                  float v1 = fy1*y1*invz1+cy1;
                  float errX1 = u1 - kp1.pt.x;
                  float errY1 = v1 - kp1.pt.y;
                  // 基于卡方检验计算出的阈值。
                  if( (errX1*errX1+errY1*errY1)>5.991*sigmaSquare1 )
                      continue;
              }

              // 双目。
              else
              {
                  float u1 = fx1*x1*invz1+cx1;
                  float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                  float v1 = fy1*y1*invz1+cy1;
                  float errX1 = u1 - kp1.pt.x;
                  float errY1 = v1 - kp1.pt.y;
                  float errX1_r = u1_r - kp1_ur;
                  if( (errX1*errX1+errY1*errY1+errX1_r*errX1_r) > 7.8*sigmaSquare1 )
                      continue;
              }

              // 计算3D点在另一帧关键帧下的重投影误差。
              const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
              const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
              const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
              const float invz2 = 1.0/z2;
              // 非双目。
              if(!bStereo2)
              {
                  float u2 = fx2*x2*invz2+cx2;
                  float v2 = fy2*y2*invz2+cy2;
                  float errX2 = u2 - kp2.pt.x;
                  float errY2 = v2 - kp2.pt.y;
                  if( (errX2*errX2+errY2*errY2)>5.991*sigmaSquare2 )
                      continue;
              }

              // 双目。
              else
              {
                  float u2 = fx2*x2*invz2+cx2;
                  float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                  float v2 = fy2*y2*invz2+cy2;
                  float errX2 = u2 - kp2.pt.x;
                  float errY2 = v2 - kp2.pt.y;
                  float errX2_r = u2_r - kp2_ur;
                  if( (errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2 )
                      continue;
              }

              // 步骤6.7 检查尺度连续性。

              // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点。
              cv::Mat normal1 = x3D-Ow1;
              float dist1 = cv::norm(normal1);

              cv::Mat normal2 = x3D-Ow2;
              float dist2 = cv::norm(normal2);

              if(dist1==0||dist2==0)
                  continue;

              // ratioDist不考虑金字塔尺度下的距离比例。
              const float ratioDist = dist2/dist1;

              // 金字塔比例因子。
              const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

              // 尺度连续判断。
              if(ratioDist*ratioFactor<ratioOctave || ratioDist > ratioOctave*ratioFactor)
                  continue;

              // 步骤6.8 三角化生成的3D点成功，构造MapPoint
              MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

              // 步骤6.9 为该MapPoint添加属性。
              // a. 观测到该MapPoint的关键帧。
              // b. 该MapPoint的描述子。
              // c. 该MapPoint的平均观测方向。
              pMP->AddObservation(mpCurrentKeyFrame, idx1);
              pMP->AddObservation(pKF2, idx2);

              mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
              pKF2->AddMapPoint(pMP, idx2);

              pMP->ComputeDistinctiveDescriptors();
              
              pMP->UpdateNormalAndDepth();

              mpMap->AddMapPoint(pMP);

              // 步骤6.9 将新生程的点放入检测队列中，通过MapPointCulling函数的检验。
              mlpRecentAddedMapPoints.push_back(pMP);

              // 新添加MapPoint数量。
              nnew++;
          } // 步骤6，三角化生成3D点。
        }   // 遍历所有关键帧。

    }



    // 检查并融合当前关键帧与相邻关键帧重复的MapPoints。
    void LocalMapping::SearchInNeighbors()
    {
        // 步骤1 获取当前关键帧在Covisible图中的权重排名前nn的邻接关键帧。
        // 找到当前帧一级相邻与耳机相邻关键帧。
        int nn = 10;
        if(mbMonocular)
            nn = 20;
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        vector<KeyFrame *> vpTargetKFs;
        for(vector<KeyFrame *>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
        {
            KeyFrame *pKFi = *vit;
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
                continue;

            // 加入相邻帧。
            vpTargetKFs.push_back(pKFi);
            // 标记加入。
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

            // 扩展到二级相邻帧，就是当前关键帧相邻关键帧在Covisible图中的邻接帧。
            const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
            for(vector<KeyFrame *>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
            {
                KeyFrame * pKFi2 = *vit2;
                if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                    continue;
                vpTargetKFs.push_back(pKFi2);
            }
        }

        ORBmatcher matcher;

        // 步骤2 将当前关键帧的MapPoint分别与一级二级邻接关键帧进行融合。
        vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for(vector<KeyFrame *>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
        {
            KeyFrame * pKFi = *vit;

            // 投影当前帧的MapPoint到pKFi中，判断是否有重复的MP。
            // 1. 如果当前帧MapPoint能匹配关键帧中的特征点，并且该点对应有MP，将两个MP融合。
            // 2. 如果当前帧MapPoint能匹配关键帧中的特征点，并且该点没有对应的MP，添加为MP。
            matcher.Fuse(pKFi, vpMapPointMatches);
        }

        // 用于存储一级邻接和二级邻接关键帧所有MapPoint的集合。
        vector<MapPoint *> vpFuseCandidates;
        vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

        // 步骤3 将一级与二级相邻关键帧的MapPoints分别与当前关键帧的MapPoint进行融合。
        // 遍历所有邻接关键帧。
        for(vector<KeyFrame *>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
        {
            KeyFrame *pKFi = *vitKF;

            vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

            // 遍历邻接关键帧中的MapPoints。
            for(vector<MapPoint *>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
            {
                MapPoint *pMP = *vitMP;
                if(!pMP)
                    continue;

                // 判断pMP是否为坏点，或者是否已经加入vpFuseCandidates。
                if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                    continue;

                // 加入集合，标记。
                pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpFuseCandidates.push_back(pMP);
            }
        }

        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

        // 步骤4 更新当前关键帧的MapPoint的描述子，深度，观测方向等属性。
        vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
        {
            MapPoint * pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    // 在所有找到pMP的关键帧中，获取最佳描述子。
                    pMP->ComputeDistinctiveDescriptors();
                    
                    // 更新平均观测方向和观测距离。
                    pMP->UpdateNormalAndDepth();
                }
            }
        }

        // 步骤5 更新当前帧与其他帧的连接关系，Covisible图。
        mpCurrentKeyFrame->UpdateConnections();
    }



    // 根据两关键帧的位姿计算两帧之间的基本矩阵F12。
    cv::Mat LocalMapping::ComputeF12(KeyFrame * &pKF1, KeyFrame * &pKF2)
    {
        // 本征矩阵E=t12 x R12;
        // 基本矩阵F=inv(K1)*E*inv(K2)。

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        // .t表示转置。
        cv::Mat R12 = R1w*R2w.t();
        cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        return K1.t().inv()*t12x*R12*K2.inv();

    }



    // 发送停止请求。
    void LocalMapping::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopRequested = true;
        unique_lock<mutex> lock2(mMutexNewKFs);
        mbAbortBA = true;
    }

    // 判断是否停止。
    bool LocalMapping::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);
        if(mbStopRequested && !mbNotStop)
        {
            mbStopped = true;
            cout << "Local Mapping STOP" << endl;
            return true;
        }

        return false;
    }

    // 表示线程停止状态。
    // true 线程停止，false 未停止。
   bool LocalMapping::isStopped()
   {
       unique_lock<mutex> lock(mMutexStop);
       return mbStopped;
   } 

   // 表示停止请求状态。
   // true 发送停止请求，fasle未发送停止请求。
   bool LocalMapping::stopRequested()
   {
       unique_lock<mutex> lock(mMutexStop);
       return mbStopRequested;
   }

   // 释放线程
   void LocalMapping::Release()
   {
       unique_lock<mutex> lock(mMutexStop);
       unique_lock<mutex> lock2(mMutexFinish);
       if(mbFinished)
           return;
       mbStopped = false;
       mbStopRequested = false;
       for(list<KeyFrame *>::iterator lit=mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
           delete *lit;
       mlNewKeyFrames.clear();

       cout << "Local Mapping RELEASE" << endl;

   }

   // 是否可以接收关键帧。
   bool LocalMapping::AcceptKeyFrames()
   {
       unique_lock<mutex> lock(mMutexAccept);
       return mbAcceptKeyFrames;
   }

   // 设置是否接收关键帧。
   void LocalMapping::SetAcceptKeyFrames(bool flag)
   {
       unique_lock<mutex> lock(mMutexAccept);
       mbAcceptKeyFrames = flag;
   }

   // 设置不停止线程。
   // 参数，true不停止线程，false停止线程。
   // return true 停止线程，false 不停止线程。
   bool LocalMapping::SetNotStop(bool flag)
   {
       unique_lock<mutex> lock(mMutexStop);

       if(flag && mbStopped)
           return false;

       mbNotStop = flag;

       return true;
   }

   // 中断BA
   void LocalMapping::InterruptBA()
   {
       mbAbortBA = true;
   }



   // 关键帧剔除。
   // 在Covisiblity图中的关键帧，其中90%以上的MapPoints能被其他至少3个关键帧观测到，则认为冗余，剔除。
   void LocalMapping::KeyFrameCulling()
   {
       // 步骤1 根据Covisiblity图提取当前帧的共视关键帧。
       vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

       // 遍历局部关键帧。
       for(vector<KeyFrame *>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
       {
           KeyFrame *pKF = *vit;
           if(pKF->mnId==0)
               continue;
           
           // 步骤2 提取每个共视关键帧的MapPoints。
           const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

           int nObs = 2;
           if(mbMonocular)
               nObs = 3;
           const int thObs=nObs;
           int nRedundantObservations=0;
           int nMPs=0;

           // 步骤3 遍历该局部关键帧的MapPoints，判断是否有90%以上的MapPoints能被其至少3个他关键帧观测到。
           for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
           {
               MapPoint *pMP = vpMapPoints[i];
               if(pMP)
               {
                   if(!pMP->isBad())
                   {
                       // 双目，只考虑近处的MapPoints，
                       if(!mbMonocular)
                       {
                           if(pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i]<0) 
                               continue;
                       }

                       nMPs++;

                       // MapPoints至少被3个关键帧观测到。
                       if(pMP->Observations() > thObs)
                       {
                           const int &scaleLevel = pKF->mvKeysUn[i].octave;
                           const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                           // 判断该MapPoint是否同时被三个尺度更好关键帧观测到。
                           int nObs=0;

                           // 遍历所有观测到MP的关键帧。
                           for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                           {
                               KeyFrame *pKFi = mit->first;
                               if(pKFi==pKF)
                                   continue;
                               const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                               // 尺度约束，要求MapPoints在关键帧pKFi的特征尺度近似于关键帧pKF的特征尺度。
                               if(scaleLeveli <= scaleLevel+1)
                               {
                                   nObs++;
                                   // 已经找到三个相同尺度的关键帧可以观测到MapPoint，不需要再找。
                                   if(nObs >= thObs)
                                       break;
                               }
                           }   // 遍历可观测到该MP的关键帧。
                           
                           // 被三个更好尺度的关键帧观测到的点云数量+1。
                           if(nObs>=thObs)
                           {
                               nRedundantObservations++;
                           }
                       }    // 至少三个关键帧观测到。
                   } // !pMP->isBad()。
               } // pMP。
           }    // 步骤3。

           // 步骤4 该局部关键帧90%以上的MapPoints能被其他关键帧（至少3个）观测到，认为冗余
           if(nRedundantObservations>0.9*nMPs)
               pKF->SetBadFlag();
       }    // 遍历共视局部关键帧。

   }



   // 生成反对称矩阵。
   cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
   {
       return (cv::Mat_<float>(3,3) <<  0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2), 0, -v.at<float>(0),
                -v.at<float>(1), v.at<float>(0), 0);

   }



   // 等待设置重置。
   // 未完成则延时等待，完成退出函数。
   void LocalMapping::RequestReset()
   {
       {
           unique_lock<mutex> lock(mMutexReset);
           mbResetRequested = true;
       }

       while(1)
       {
           {
               unique_lock<mutex> lock2(mMutexReset);
               if(!mbResetRequested)
                   break;
           }

           // 延时3000us
           std::this_thread::sleep_for(std::chrono::milliseconds(3));
       }

   }

   // 设置重置，清空关键帧队列和点云队列。
   void LocalMapping::ResetIfRequested()
   {
       unique_lock<mutex> lock(mMutexReset);
       if(mbResetRequested)
       {
           mlNewKeyFrames.clear();
           mlpRecentAddedMapPoints.clear();
           mbResetRequested=false;
       }
   }

   // 重置完成。
   void LocalMapping::RequestFinish()
   {
       unique_lock<mutex> lock(mMutexFinish);
       mbFinishRequested = true;
   }

   // 线程是否完成。
   bool LocalMapping::CheckFinish()
   {
       unique_lock<mutex> lock(mMutexFinish);
       return mbFinishRequested;
   }

   // 设置完成和结束标志位。
   void LocalMapping::SetFinish()
   {
       unique_lock<mutex> lock(mMutexFinish);
       mbFinished = true;
       unique_lock<mutex> lock2(mMutexStop);
       mbStopped = true;
   }

   // 返回线程完成状态。
   bool LocalMapping::isFinished()
   {
       unique_lock<mutex> lock(mMutexFinish);
       return mbFinished;
   }



















}   // namespace ORB_SLAM2


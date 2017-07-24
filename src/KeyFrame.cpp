


#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"

#include <mutex>


namespace ORB_SLAM2
{
    // 静态成员变量定义。
    long unsigned int KeyFrame::nNextId=0;

    // 构造函数。
    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), 
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), 
    im_(F.im_.clone()),rgb_(F.rgb_.clone()), semidense_flag_(false)
    {
        mnId = nNextId++;

        // 传递每个栅格的特征点数。
        mGrid.resize(mnGridCols);
        for(int i=0; i<mnGridCols; i++)
        {
            mGrid[i].resize(mnGridRows);
            for(int j=0; j<mnGridRows; j++)
                mGrid[i][j] = F.mGrid[i][j];
        }

        SetPose(F.mTcw);
	

	// 半稠密参数初始化。
	cv::Mat image_mean,image_stddev,gradx,grady;
	cv::meanStdDev(im_,image_mean,image_stddev);
	I_stddev = image_stddev.at<double>(0,0);
	image_mean.~Mat();
	image_stddev.~Mat();

	gradx = cv::Mat::zeros(im_.rows, im_.cols, CV_32F);
	grady = cv::Mat::zeros(im_.rows, im_.cols, CV_32F);
	cv::Scharr(im_, gradx, CV_32F, 1, 0, 1/32.0);
	cv::Scharr(im_, grady, CV_32F, 0, 1, 1/32.0);
	cv::magnitude(gradx,grady,GradImg);
	cv::phase(gradx,grady,GradTheta,true);
	gradx.~Mat();
	grady.~Mat();

	// 7.8%
	
	depth_map_ = cv::Mat::zeros(im_.rows, im_.cols, CV_32F);
	depth_sigma_ = cv::Mat::zeros(im_.rows, im_.cols, CV_32F);
	SemiDensePointSets_ = cv::Mat::zeros(im_.rows, im_.cols, CV_32FC3);
	// 33.2%

      
	// std::cout << "mc5" <<endl;
    }



    // 计算mBowVec，并且将描述子分散在第4层，即mFeatVec记录了属于第i个node的ni个描述子。
    void KeyFrame::ComputeBoW()
    {
        if(mBowVec.empty() || mFeatVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }

    }



    // 设置位姿。
    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
        unique_lock<mutex> lock(mMutexPose);
        Tcw_.copyTo(Tcw);
        cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Tcw.rowRange(0,3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc*tcw;

        Twc = cv::Mat::eye(4,4,Tcw.type());
        Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
        Ow.copyTo(Twc.rowRange(0,3).col(3));

        // 相机坐标系下(左目)，立体相机中心的齐次坐标。
        cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0, 0, 1);
        // 世界坐标系下，立体相机中心的齐次坐标。
        Cw = Twc*center;

    }



    // 获取位姿。
    cv::Mat KeyFrame::GetPose()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.clone();

    }



    // 获取位姿的逆。
    cv::Mat KeyFrame::GetPoseInverse()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Twc.clone();

    }



    // 获取世界坐标系下相机坐标系原点坐标。
    cv::Mat KeyFrame::GetCameraCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Ow.clone();

    }



    // 获取双目相机中点坐标(世界坐标系下)。
    cv::Mat KeyFrame::GetStereoCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Cw.clone();

    }



    // 获取当前关键帧的旋转矩阵。
    cv::Mat KeyFrame::GetRotation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0,3).colRange(0,3).clone();

    }



    // 获取当前关键帧帧的位移矩阵。
    cv::Mat KeyFrame::GetTranslation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0,3).col(3).clone();

    }



    /* 为关键帧之间添加连接。
    *@param
    *  pKF          关键帧。
    *  weight       权重，该关键帧与pKF共同观测到的3D点数量。

    *  更新mConnectedKeyFrameWeights
    */ 
    void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            // std::map::count 只能返回0或1。
            // 返回0表示mConnectedKeyFrameWeights中没有pKF,没有相连。
            // 返回1表示相连。
            if(!mConnectedKeyFrameWeights.count(pKF))
                mConnectedKeyFrameWeights[pKF] = weight;
            else if(mConnectedKeyFrameWeights[pKF] != weight)
                mConnectedKeyFrameWeights[pKF] = weight;
            else
                return;
        }

        UpdateBestCovisibles();

    }

    // 按照权重对连接的关键帧进行排序。
    // 更新后的结果保存在mvpOrderedConnectedKeyFrames和mvOrderedWeights中。
    void KeyFrame::UpdateBestCovisibles()
    {
        unique_lock<mutex> lock(mMutexConnections);
        //http://stackoverflow.com/questions/3389648/difference-between-stdliststdpair-and-stdmap-in-c-stl
        vector<pair<int, KeyFrame*>> vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());

        // 取出所有连接的关键帧，mConnectedKeyFrameWeights的类型为std::map<KeyFrame*, int>, 
        // 而vpairs将共视权重放在前面，方便排序。
        for(map<KeyFrame *, int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
            vPairs.push_back(make_pair(mit->second, mit->first));

        // 按照权重进行排序。 
        sort(vPairs.begin(), vPairs.end());
        list<KeyFrame *> lKFs;
        list<int> lWs;
        for(size_t i=0, iend=vPairs.size(); i<iend; i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        // 迭代器初始化vector对象，并赋值。
        mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

    }



    // 获得与该关键帧相连的关键帧。
    // @return 连接的关键帧。
    set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        set<KeyFrame *> s;
        for(map<KeyFrame *, int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end();mit!=mend; mit++)
           s.insert(mit->first); 
        return s;

    }



    // 获得与当前关键帧相连的关键帧(以按权重排序)。
    // @return 连接的关键帧。
    vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;

    }



    // 获得与当前关键帧相连的前N个关键帧(按权值排序)。
    // 如果少于N个，返回所有关键帧。
    vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
        unique_lock<mutex> lock(mMutexConnections);
        if((int)mvpOrderedConnectedKeyFrames.size()<N)
            return mvpOrderedConnectedKeyFrames;
        else
            return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+N);

    }



    // 获得与当前关键帧连接的权重>=W的关键帧。
    vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
        unique_lock<mutex> lock(mMutexConnections);

        if(mvpOrderedConnectedKeyFrames.empty())
            return vector<KeyFrame *>();

        // 从mvOrderedWeights找出第一个大于w的迭代器。
        vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
        // 不存在。
        if(it==mvOrderedWeights.end())
            return vector<KeyFrame *>();
        else
        {
            int n = it - mvOrderedWeights.begin();
            return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
        }

    }



    // 获得当前关键帧与pKF的权重。
    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
            return mConnectedKeyFrameWeights[pKF];
        else
            return 0;

    }



    // 为该关键帧添加地图点pMP，索引idx。
    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    // 剔除关键帧中的地图点。
    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
    {
        int idx = pMP->GetIndexInKeyFrame(this);
        if(idx>=0)
            mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);

    }



    // 替换该关键帧地图点。
    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
    {
        mvpMapPoints[idx] = pMP;
    }



    // 获得该关键帧所有地图点。
    set<MapPoint *> KeyFrame::GetMapPoints()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPoint *>s;
        for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
        {
            if(!mvpMapPoints[i])
                continue;

            MapPoint *pMP = mvpMapPoints[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;

    }



    /* 
    *  获得该关键帧中高质量的点云数量。
    *  minObs是一个阈值，大于minObs表示是高质量的MapPoint。 
    *  一个高质量的MapPoint会被多个KeyFrame观测到。
    *  return 高质量关键帧数量。
    */
    int KeyFrame::TrackedMapPoints(const int &minObs)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        int nPoints = 0;
        const bool bCheckObs = minObs>0;
        for(int i=0; i<N; i++)
        {
            MapPoint *pMP = mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(bCheckObs)
                    {
                        // 判断地图点质量。
                        if(mvpMapPoints[i]->Observations() >= minObs)
                            nPoints++;
                    }
                    else
                        nPoints++;
                }
            }
        }

        return nPoints;

    }



    // 获取该关键帧的所有MapPoints。
    vector<MapPoint *> KeyFrame::GetMapPointMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints;

    }

    // 获取该关键帧编号为idx的地图点。
    MapPoint * KeyFrame::GetMapPoint(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }



    /*
    *   更新Covisibility图的连接关系。
    *   1. 获得关键帧的所有MapPoints，统计观测到这些3D点的关键帧与其他关键帧的共视程度。
    *      对于每一个找到的关键帧，建立一条边，权重是共视3D点的个数。
    *   2. 且该权重需要大于一个阈值，如果没有，只保留权重最大的边。
    *   3. 对于这些权重，从大到小排序，便于处理。
    *   更新完Covisibility图，如果没有初始化过，初始化为连接权重最大的边。
    */
    void KeyFrame::UpdateConnections()
    {
        // 执行UpdateConnection()之前，关键帧只与MapPoints有连接，该函数将关键帧连接起来。


        /*****1****/
        // 存储与当前关键帧 存在共视关系的关键帧-共视权重。
        map<KeyFrame*, int> KFcounter;

        vector<MapPoint *> vpMP;

        // 提取关键帧的所有MapPoints。
        {
            unique_lock<mutex> lockMPs(mMutexFeatures);
            vpMP = mvpMapPoints;
        }

        // 通过3D统计可以观测到这些3D点的所有关键帧之间的共视程度。
        // 即统计每一个关键帧都有多少关键帧与它存在共视关系，统计结果存储在KFcounter。
        for(vector<MapPoint *>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
        {
            MapPoint *pMP = *vit;

            if(!pMP)
                continue;

            if(pMP->isBad())
                continue;

            // 对于每一个MapPoint，observations记录了可以观测到该MapPoint的所有关键帧和KF中Mappoint的索引。
            map<KeyFrame*, size_t> observations = pMP->GetObservations();

            for(map<KeyFrame*, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                // 剔除自身，自身不算共视。
                if(mit->first->mnId==mnId)
                    continue;
                KFcounter[mit->first]++;
            }
        }

        // 正常不因该出现。
        if(KFcounter.empty())
            return;



        /*****2******/
       
        int nmax = 0;
        KeyFrame *pKFmax = NULL;
        // 共视阈值。
        int th =15;

        // vPairs记录了与其他关键帧共视权重大于th的关键帧。
        // pair<int, KeyFrame *>权重写在前面，方便计算。
        vector<pair<int, KeyFrame *> > vPairs;
        vPairs.reserve(KFcounter.size());
        for(map<KeyFrame *, int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
        {
            // 找到对应最大权重的关键帧。
            if(mit->second > nmax)
            {
                nmax = mit->second;
                pKFmax = mit->first;
            }

            // 权重>阈值。
            if(mit->second >= th)
            {

                vPairs.push_back(make_pair(mit->second, mit->first));
                // 更新KFcounter中的关键帧与当前关键帧的连接权重，mConnectedKeyFrameWeights。
                (mit->first)->AddConnection(this, mit->second);
            }
        }

        // 与当前关键帧共视的关键帧没有大于共视阈值的，对权重最大的关键帧建立连接。
        if(vPairs.empty())
        {
            vPairs.push_back(make_pair(nmax, pKFmax));
            pKFmax->AddConnection(this, nmax);
        }

        // 对vPairs按权重从大到小排序。
        sort(vPairs.begin(), vPairs.end());
        list<KeyFrame *> lKFs;
        list<int> lWs;
        for(size_t i=0; i<vPairs.size(); i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        /*****3*******/
        {
            unique_lock<mutex> lockCon(mMutexConnections);

            // 更新图的连接。
            mConnectedKeyFrameWeights = KFcounter;      // 更新该KeyFrame的mConnectedKeyFrameWeights，更新共视关键帧与共视权重。
            mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());    // 更新添加连接关系的关键帧(排序)。
            mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());         // 更新对应共视关键帧的权重(已排序)。


            // 更新生成树的连接。
            if(mbFirstConnection && mnId != 0)
            {
                // 初始化该关键帧的父关键帧为共视程度最高的关键帧。
                mpParent = mvpOrderedConnectedKeyFrames.front();
                // 建立双向连接。
                mpParent->AddChild(this);
                mbFirstConnection = false;
            }
        }

    }



    // 最小生成数 spanning tree。
    
    // 添加子关键帧。 
    void KeyFrame::AddChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    // 剔除子关键帧。
    void KeyFrame::EraseChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.erase(pKF);
    }

    // 改变父关键帧。 
    void KeyFrame::ChangeParent(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mpParent = pKF;
        pKF->AddChild(this);
    }

    // 获取子关键帧。
    set<KeyFrame *> KeyFrame::GetChilds()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens;
    }

    // 获取父关键帧。
    KeyFrame * KeyFrame::GetParent()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    // 查找子关键帧，返回1表示有。
    bool KeyFrame::hasChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens.count(pKF);
    }

    // 添加闭环边。
    void KeyFrame::AddLoopEdge(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mbNotErase = true;
        mspLoopEdges.insert(pKF);
    }

    // 获得该帧的闭环边。
    set<KeyFrame *> KeyFrame::GetLoopEdges()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    // 设置关键帧不要被剔除。
    void KeyFrame::SetNotErase()
    {
        unique_lock<mutex> lock(mMutexConnections);
        mbNotErase = true;
    }

    // 设置擦除关键帧。
    void KeyFrame::SetErase()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mspLoopEdges.empty())
            {
                mbNotErase = false;
            }
        }

        // mbToBeErased==true 表示之前不许擦除，现在再验证一次是否进行擦除。
        if(mbToBeErased)
        {
            SetBadFlag();
        }
    }



    // 设置当前关键帧为坏帧。
    void KeyFrame::SetBadFlag()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mnId == 0)
                return;
            else if(mbNotErase)
            {
                // 只设置了标志为，实际并没有擦除。
                mbToBeErased = true;
                return;
            }
        }

        // 1. 擦除其他关键帧与当前帧的连接。
        for(map<KeyFrame *, int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
            mit->first->EraseConnection(this);

        // 2. 擦除当前帧对点云的观察关系。
        for(size_t i=0; i<mvpMapPoints.size(); i++)
            if(mvpMapPoints[i])
                mvpMapPoints[i]->EraseObservation(this);

        {
            unique_lock<mutex> lock(mMutexConnections);
            unique_lock<mutex> lock1(mMutexFeatures);

            // 3. 清空当前帧与其他关键帧之间的联系。
            mConnectedKeyFrameWeights.clear();
            mvpOrderedConnectedKeyFrames.clear();

            // 4. 更新生成树。
            set<KeyFrame *> sParentCandidates;
            sParentCandidates.insert(mpParent);

            // 如果当前关键有子关键帧，告诉这些子关键帧，父关键帧GG了，换人吧。
            while(!mspChildrens.empty())
            {
                bool bContinue = false;

                int max = -1;
                KeyFrame *pC;
                KeyFrame *pP;

                // 遍历所有子关键帧，更新他们的父关键帧。
                for(set<KeyFrame *>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
                {
                    KeyFrame* pKF = *sit;
                    if(pKF->isBad())
                        continue;


                    // 遍历每一个子关键帧的所有共视关键帧。
                    vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames(); 
                    for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                    {
                        for(set<KeyFrame *>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end();spcit!=spcend; spcit++)
                        {
                            // 当前关键帧的父关键帧是否与它的子关键帧存在共视连接关系。
                            if(vpConnected[i]->mnId == (*spcit)->mnId)
                            {
                                // 获取当前关键帧的子关键帧的共视关键帧的权重。
                                int w = pKF->GetWeight(vpConnected[i]);
                                // 找到与所有mspChildrens中的子关键帧共视的最高权重的关键帧。
                                if(w>max)
                                {
                                    pC = pKF;
                                    pP = vpConnected[i];
                                    max = w;
                                    bContinue = true;
                                }
                            }
                        }
                    }
                }

                // 设置与当前关键帧的子关键帧具有最高共视程度的关键帧 作为子关键帧的父关键帧。
                if(bContinue)
                {
                    // 因为父关键帧GG了，子关键帧更新新的福关键帧。
                    pC->ChangeParent(pP);
                    // 子关键帧找到了新的父关键帧，子关键帧升级作为其他子关键帧的备选父关键帧。
                    sParentCandidates.insert(pC);
                    // 该子节点处理完毕。
                    mspChildrens.erase(pC);
                }
                else
                    break;
            }

            // 如果还有子关键帧没有找到父关键帧。
            // 直接把父关键帧的父关键帧作为自己的父关键帧。
            if(!mspChildrens.empty())
                for(set<KeyFrame *>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
                {
                    (*sit)->ChangeParent(mpParent);
                }

            mpParent->EraseChild(this);
            mTcp = Tcw*mpParent->GetPoseInverse();
            mbBad = true;
        }

        mpMap->EraseKeyFrame(this);
        mpKeyFrameDB->erase(this); 


    }

    

    // 返回关键帧质量标志mbBad。SetBadFlag()进行了设置。
    bool KeyFrame::isBad()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mbBad;
    }



    // 擦除与pKF的连接关系。
    void KeyFrame::EraseConnection(KeyFrame *pKF)
    {
        bool bUpdate =false;
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mConnectedKeyFrameWeights.count(pKF))
            {
                mConnectedKeyFrameWeights.erase(pKF);
                bUpdate = true;
            }
        }

        // 更新连接关系排序。
        if(bUpdate)
            UpdateBestCovisibles();

    }



    // 获取该关键帧中制定区域的特征。
    vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const 
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        // floor向下取整，mfGridElementWidthInv为每个像素在横坐标占多少格子。
        const int nMinCellX = max(0, (int)floor((x-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX >= mnGridCols)
            return vIndices;

        // ceil向上取整。
        const int nMaxCellX = min((int)mnGridCols-1, (int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX < 0)
            return vIndices;
        
        // mfGridElementHeightInv为每个像素在纵坐标占多少个格子
        const int nMinCellY = max(0, (int)floor((y-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY >= mnGridRows)
            return vIndices;

        const int nMaxCellY = min((int)mnGridRows-1, (int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY < 0)
            return vIndices;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                // mGrid[ix][iy]存储着每个格子的特征点数。
                const vector<size_t> vCell = mGrid[ix][iy];
                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if(fabs(distx)<r && fabs(disty)<r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;

    }



    // 像素点是否在图像内。
    bool KeyFrame::IsInImage(const float &x, const float &y) const 
    {
        return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
    }



    /**
     * 立体相机。
     * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
     * @param  i 第i个keypoint
     * @return   3D点（相对于世界坐标系）
     */
    cv::Mat KeyFrame::UnprojectStereo(int i)
    {
        const float z = mvDepth[i];
        if(z>0)
        {
            // 由2维图像反投影到相机坐标系
            // mvDepth是在ComputeStereoMatches函数中求取的
            // mvDepth对应的校正前的特征点，因此这里对校正前特征点反投影
            // 可在Frame::UnprojectStereo中却是对校正后的特征点mvKeysUn反投影
            // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
            const float u = mvKeys[i].pt.x;
            const float v = mvKeys[i].pt.y;
            const float x = (u-cx)*z*invfx;
            const float y = (v-cy)*z*invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

            unique_lock<mutex> lock(mMutexPose);
            // 由相机坐标系转换到世界坐标系
            // Twc为相机坐标系到世界坐标系的变换矩阵
            // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
            return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
        }
        else
            return cv::Mat();
    }



    /*
    *   评估当前关键帧场景深度，q=2表示中值。
    *   q q=2;
    *   return 中值深度。
    */
    float KeyFrame::ComputeSceneMedianDepth(const int q)
    {
        vector<MapPoint *> vpMapPoints;
        cv::Mat Tcw_;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPose);
            vpMapPoints = mvpMapPoints;
            Tcw_ = Tcw.clone();
        }

        vector<float> vDepths;
        vDepths.reserve(N);
        cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
        Rcw2 = Rcw2.t();
        float zcw = Tcw_.at<float>(2,3);
        for(int i=0; i<N; i++)
        {
            if(mvpMapPoints[i])
            {
                MapPoint *pMP = mvpMapPoints[i];
                cv::Mat x3Dw = pMP->GetWorldPos();
                // R*x3Dw+t的第三行。
                float z = Rcw2.dot(x3Dw)+zcw;
                vDepths.push_back(z);
            }
        }
        
        sort(vDepths.begin(), vDepths.end());

        return vDepths[(vDepths.size()-1)/q];    
	}

	

	// 半稠密函数。 
	cv::Mat KeyFrame::GetImage()
	{
	  return im_.clone();
	}




	cv::Mat KeyFrame::GetCalibrationMatrix() const
	{
	  return mK.clone();
	}
	
	

	vector<float> KeyFrame::GetAllPointDepths(int q)
	{
	  vector<MapPoint*> vpMapPoints;
	  cv::Mat Tcw_;
	  
	  {
	    unique_lock<mutex>  lock(mMutexFeatures);
	    unique_lock<mutex>  lock2(mMutexPose);
	    vpMapPoints = mvpMapPoints;
	    Tcw_ = Tcw.clone();
	  }

	  vector<float> vDepths;
	  vDepths.reserve(mvpMapPoints.size());
	  cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
	  Rcw2 = Rcw2.t();
	  float zcw = Tcw_.at<float>(2,3);
	  for(size_t i=0; i<mvpMapPoints.size(); i++)
	  {
	    if(mvpMapPoints[i])
	    {
	      MapPoint* pMP = mvpMapPoints[i];
	      cv::Mat x3Dw = pMP->GetWorldPos();
	      float z = Rcw2.dot(x3Dw)+zcw;
	      vDepths.push_back(z);
	      
	    }
	    
	  }

	  sort(vDepths.begin(),vDepths.end());
	  
	  return vDepths;
	  
	}
	
	

	
}   // namespace ORB_SLAM2













#include "LoopClosing.h"
#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"

#include <mutex>
#include <thread>



namespace ORB_SLAM2
{

    // 构造函数，初始化成员变量。
    LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
        mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
        mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
        mbStopGBA(false), mbFixScale(bFixScale)
    {
        mnCovisibilityConsistencyTh = 3;
        mpMatchedKF = NULL;
    }

    // 设置线程间对象的指针变量。
    void LoopClosing::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }
    void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    // 线程主函数。
    void LoopClosing::Run()
    {
        mbFinished = false;

        while(1)
        {
            // 检测LocalMapping发来的关键帧队列mlpLoopKeyFrameQueue是否为空。
            if(CheckNewKeyFrames())
            {
                // 闭环候选检测，检查Covisiblity图的一致性。
                if(DetectLoop())
                {
                    // 计算相似变换，[sR|t]。双目和RGBD中s=1。
                    if(ComputeSim3())
                    {
                        // 运行闭环融合和pose图优化。
                        CorrectLoop();
                    }
                }
            }

            // 如果收到重置申请，进行重置。
            ResetIfRequested();
            
            if(CheckFinish())
                break;
            // 延时5000us。
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        }

        // 设置线程完成标志位。
        SetFinish();
        
    }



    // 插入LocalMapPing线程得到的关键帧。
    void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        if(pKF->mnId != 0)
            mlpLoopKeyFrameQueue.push_back(pKF);
    }



    // 查看队列中是否有关键帧。
    // return true 表示存在，false表示空。
    bool LoopClosing::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        return (!mlpLoopKeyFrameQueue.empty());
    }



    // 闭环候选帧检测，获取闭环帧。
    // 当前关键帧的闭环帧的covisibility图与之前三个关键帧的covisibility图有共同交点(不需要必须相同，可以是不同的3个交点）。
    bool LoopClosing::DetectLoop()
    {
        {
            // 从队列中取出一个关键帧。
            unique_lock<mutex> lock(mMutexLoopQueue);
            mpCurrentKF = mlpLoopKeyFrameQueue.front();
            mlpLoopKeyFrameQueue.pop_front();
            
            // 避免线程处理过程中，关键帧被擦除。
            mpCurrentKF->SetNotErase();
        }

        // 步骤1 如果距离上次闭环没多久(<10帧)，或者Map中总共都没有10帧，就不闭环了。
        if(mpCurrentKF->mnId < mLastLoopKFid+10)
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mpCurrentKF->SetErase();
            return false;
        }

        // 步骤2 遍历所有共视关键帧，计算当前关键帧与每个共视关键帧的BoW相似得分，得到最低分minScore。
        const vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
        const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
        float minScore = 1;
        for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
        {
            KeyFrame *pKF = vpConnectedKeyFrames[i];
            if(pKF->isBad())
                continue;
            const DBoW2::BowVector &BowVec = pKF->mBowVec;

            float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

            if(score < minScore)
                minScore = score;
        }

        // 步骤3 在KFDB中找出当前关键CuKF的闭环备选帧，相似性大于最低分。
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

        // 如果没有闭环候选帧，把新的关键帧添加到KFDB中并且返回false。
        // 没有发生闭环的时候，vpCandidateKFs一直是找不到的，mvConsistentGroups会被清空。
        if(vpCandidateKFs.empty())
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mvConsistentGroups.clear();
            mpCurrentKF->SetErase();
            return false;
        }

        // 步骤4 在候选帧中检测具有一致性的候选帧。
        // 1. 每个候选帧与其相连的关键帧构成了一个Covisiblity图的子候选组vpCandidateKFs->spCandidateGroup。
        // 2. 检测子候选组中每一个关键帧是否在一致组中。
        //    如果在一致组，nCurrentConsisitency=nPreviousConsistency+1。并把候选组放入一致组vCurrentConsistentGroups。
        //    如果不在一致组， 当前候选组插入到vCurrentConsistentGroups的第一个对元素中(0)。
        // 3. 如果nCurrentConsistency < 3，当前候选帧的候选组加入到vCurrentConsistentGroups。 
        //    如果nCurrentConsistency >=3，那么该子候选组的候选帧是闭环候选帧，加入mvpEnoughConsistentCandidates，
        //    当前候选组也加入vCurrentConsistentGroups。

        mvpEnoughConsistentCandidates.clear();  // 最终筛选的闭环帧。

        // ConsistentGroup 类型是pari<set<KeyFrame *>, int> ,第一个表示每个一致组的关键帧，第二个时每个一致组的序号。
        vector<ConsistentGroup> vCurrentConsistentGroups;
        // 所有子一致组的标识符都是false。
        vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
        for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
        {
            KeyFrame * pCandidateKF = vpCandidateKFs[i];

            // 将自己以及自己相连的关键帧构成一个子候选组。
            set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
            spCandidateGroup.insert(pCandidateKF);

            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;

            // 遍历之前的子一致组。
            for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
            {
                // 取出一个之前的子一致组。
                set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

                // 遍历每个子候选组的关键帧，检测子候选组中每个关键帧在子一致组中是否存在。
                // 如果存在，则子候选组 与 该子一致组一致。
                bool bConsistent = false;
                for(set<KeyFrame *>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send; sit++)
                {
                    // sPreviousGroup.count(*sit)返回子一致组中包含关键帧*sit(子候选关键组的)的数量。
                    if(sPreviousGroup.count(*sit))
                    {
                        bConsistent = true; // 该子候选组与该子一致组一致。
                        bConsistentForSomeGroup = true; // 该子候选组最少与一个子一致组相连。
                        break;
                    }
                }

                if(bConsistent)
                {
                    int nPreviousConsistency = mvConsistentGroups[iG].second;   //之前子一致组的序号。
                    int nCurrentConsistency = nPreviousConsistency+1;           // 当前子候选组的序号。
                    
                    if(!vbConsistentGroup[iG])
                    {
                        // 将该子候选组的打上编号加入当前一致组。
                        ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                        vCurrentConsistentGroups.push_back(cg);
                        vbConsistentGroup[iG]=true; // 添加之后，避免重复添加，标志为true。
                    }
                    
                    // 满足要求，添加候选帧。
                    if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                    {
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                        bEnoughConsistent=true; 
                    }
                }
            }

            // 如果该子候选组的所有关键帧都不存在于子一致组中，vCurrentConsistentGroups为空。
            // 把子候选组全部拷贝到vCurrentConsistentGroups，最终更新mvConsistentGroups， 计数器清0，重新开始。
            if(!bConsistentForSomeGroup)
            {
                ConsistentGroup cg = make_pair(spCandidateGroup, 0);
                vCurrentConsistentGroups.push_back(cg);
            }
        }

        // 更新Covisibility 一致组。
        mvConsistentGroups = vCurrentConsistentGroups;

        // 将当前关键帧添加到KFDB。
        mpKeyFrameDB->add(mpCurrentKF);

        // 没有找到候选帧。
        if(mvpEnoughConsistentCandidates.empty())
        {
            mpCurrentKF->SetErase();
            return false;
        }
        else
        {
            return true;
        }

        mpCurrentKF->SetErase();
        return false;
    }



    // 计算相似变换。 
    bool LoopClosing::ComputeSim3()
    {
        // 对于每一个一致闭环，计算相似变换sim3。

        // 获取闭环候选帧数量。
        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // 首先对候选帧进行ORB特征匹配，如果有足够的匹配，设置sim3求解器。
        ORBmatcher matcher(0.75, true);

        vector<Sim3Solver *> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);   // 为每一个候选帧建立一个sim3求解器。
        
        vector<vector<MapPoint *> > vvpMapPointMatches;     // 存放每一个候选帧的匹配。
        vvpMapPointMatches.resize(nInitialCandidates);

        // 剔除标志位。
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        int nCandidates = 0;    // 有足够匹配的候选帧数量。

        for(int i=0; i<nInitialCandidates; i++)
        {
            // 步骤1 从筛选的闭环帧中选取一帧关键帧pKF。
            KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

            // 防止在LocalMapPing中KeyFrameCulling函数将次关键帧作为冗余关键帧剔除。
            pKF->SetNotErase();

            if(pKF->isBad())
            {
                // 直接舍弃。
                vbDiscarded[i] = true;
                continue;
            }

            // 步骤2 将当前帧mpCurrentKF与闭环候选关键帧pKF匹配。
            // 通过BoW加速得到mpCurrentKF与pKF之间的匹配特征点，存在vvpMapPointMatches中。
            int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

            // 匹配数量太少，直接剔除。
            if(nmatches < 20)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 构造sim3求解器
                // 如果mbFixScale=true，是6DoF优化(双目, RGBD)，如果是false，7DoF优化(单目)。
                Sim3Solver *pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
                pSolver->SetRansacParameters(0.99, 20, 300);    // 至少20个内点，300次迭代。
                vpSim3Solvers[i] = pSolver;
            }

            // 参与sim3计算的候选关键帧+1。
            nCandidates++;
        }

        bool bMatch =false;     // 用于标记是否有一个候选帧通过sim3的求解与优化。

        // 循环所有候选帧，每个候选帧迭代5次，如果5次以后得不到结果，换下一帧候选帧。
        // 直到有一个候选帧首次迭代成功(bMatch=true)，或者某个候选帧迭总的迭代次数超限，剔除直到没有可用候选帧。
        while(!bMatch && nCandidates>0)
        {
            for(int i=0; i<nInitialCandidates; i++)
            {
                if(vbDiscarded[i])
                    continue;

                KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

                // 运行5次RANSAC迭代。
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;       // 无需复制，iterate函数会进行赋值初始化操作。

                // 步骤3 对步骤2中有较好匹配的候选关键帧进行Sim3变换。
                Sim3Solver *pSolver = vpSim3Solvers[i];
                // 最多迭代5次，返回的Scm时候选帧pKF到当前帧mpCurrentKF的Sim3变换。
                cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // 经过n次循环，每次迭代5次，共迭代n*5次。
                // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除。
                if(bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // 如果RANSAC返回一个Sim3矩阵，进行引导匹配并优化所有关联关系。
                if(!Scm.empty())
                {
                    vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint *> (NULL));
                    for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                    {
                        // 保存候选帧inlier的MapPoint
                        if(vbInliers[j])
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                    }

                    // 步骤4 通过步骤3求取的Sim3变换引导关键帧匹配弥补步骤2中的漏匹配。
                    // sR t
                    // 0  1
                    cv::Mat R = pSolver->GetEstimatedRotation();    // 候选关键帧pKF到当前帧mpCurrentKF的R(R12)。
                    cv::Mat t = pSolver->GetEstimatedTranslation(); // 候选关键帧pKF到当前帧mpCurrentKF的t(t12)，当前帧坐标系下，由pKF指向当前帧。
                    const float s = pSolver->GetEstimatedScale();   // 候选关键帧pKF到当前帧mpCurrentKF的尺度变换s(s12)。

                    // 查找更过匹配(成功的闭环匹配需要很多的特征点数，之前BoW搜索会有漏匹配)。
                    // 通过Sim3变换，确定候选关键帧与当前帧的特征点的区域。
                    // 在该区域通过描述子进行匹配，捕获漏匹配点，更新匹配关系vpMapPointMatches。
                    matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

                    // 步骤5 通过Sim3优化，只要有一个候选帧通过Sim3的求解和优化，就跳出循环停止对其候选帧的判断。

                    // Mat矩阵转换为Eigen的Matrix。
                    g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
                    
                    // 如果mbFixScale为true，6DoF优化，否则为7DoF。
                    // 优化mpCurrentKF与pKF对应的MapPoint间的Sim3,得到优化后的gScm。
                    const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);   // 卡方chi2检验阈值。

                    // 如果优化成功，停止ransac并且继续。
                    if(nInliers >= 20)
                    {
                        bMatch = true;

                        // 得到最终的闭环检测出来与当前帧形成闭环的关键帧mpMatchedKF。
                        mpMatchedKF = pKF;

                        // 得到从世界坐标系到该闭环帧的Sim3变换，Scale=1。
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);

                        // 得到g2o优化后从世界坐标系到当前关键帧mpCurrentKF的sim3变换。
                        mg2oScw = gScm*gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        // 优化后的匹配内点。
                        mvpCurrentMatchedPoints = vpMapPointMatches;

                        // 只要有一个候选帧通过Sim3求解和优化，跳出对其他候选的判断。
                        break; 
                    }
                }   // !Scm.empty()
            }   // for i<nInitialCandidates
        }   // while


        // 没有一个闭环匹配候选帧通过Sim3的求解与优化。
        if(!bMatch)
        {
            // 清空mvpEnoughConsistentCandidates。
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

        // 步骤6 取出闭环匹配上关键帧mpMatchedKF的相连关键帧和点云MP。
        // 将mpMatchedKF的相邻关键帧全部去除放入vpLoopConnectedKFs。
        // 将vpLoopConnectedKFs的MapPoints全部放入mvpLoopMapPoints。
        vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
        // 包括闭环帧本身。
        vpLoopConnectedKFs.push_back(mpMatchedKF);
        mvpLoopMapPoints.clear();
        for(vector<KeyFrame *>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
        {
            KeyFrame *pKF = *vit;
            vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
            for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
            {
                MapPoint * pMP = vpMapPoints[i];
                if(pMP)
                {
                    if(!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                    {
                        mvpLoopMapPoints.push_back(pMP);
                        // 标记该MapPoints被mpCurrentKF闭环检测时添加过，避免重复添加。
                        pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                    }
                }
            }
        }

        // 步骤7 将闭环匹配关键帧和它相邻的关键帧的MapPoints投影到当前关键帧进行投影匹配。
        //       根据匹配投影查找更多的匹配。
        //       根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，根据尺度确定搜索区域，搜索范围系数=10。
        //       根据MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差<TH_LOW则成功，更新mvpCurrentMatchedPoints。
        //       mvpCurrentMatchedPoints用于SearchAndFusez中检测当前关键帧MapPoints与匹配的MapPoints是否存在冲突。
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

        // 步骤8 判断当前关键帧与检测出的所有闭环关键帧是否有足够的MapPoints匹配。
        int nTotalMatches = 0;
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
                nTotalMatches++;
        }

        // 步骤9 清空mvpEnoughConsistentCandidates。
        if(nTotalMatches >= 40)
        {
            for(int i=0; i<nInitialCandidates; i++)
                if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                    mvpEnoughConsistentCandidates[i]->SetErase();

            return true;
        }
        else
        {
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

    }



    //  闭环校正。 
    void LoopClosing::CorrectLoop()
    {
        cout << "Loop detected!" << endl;

        // 请求局部地图停止，防止局部地图线程中InsertKeyFrame()插入新的关键帧。
        mpLocalMapper->RequestStop();


        // 如果全局BA正在运行，终止。
        if(isRunningGBA())
        {
            // GBA状态标志符。
            mbStopGBA = true;

            // 等待当前GBA完成。
            while(!isFinishedGBA())
            {
                // 延时5000us。
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            // 等待mpThreadGBA执行完毕返回。
            mpThreadGBA->join();
            delete mpThreadGBA;
        }

        // 等待局部地图停止。
        while(!mpLocalMapper->isStopped())
        {
            // 延时1000us。
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // 步骤1 根据共视关系更新当前帧与其他关键帧之间的连接。
        mpCurrentKF->UpdateConnections();

        // 步骤2 通过位姿传递，得到Sim3优化后与当前关键帧相连的其他关键帧的位姿，以及他们的MapPoints。
        // 当前关键帧帧与世界坐标系之间的sim3变换已经确定并优化(ComputeSim3)。
        // 通过位姿传递，可以确定这些相连关键帧与世界坐标系之间的sim3变换。
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;

        // 保存mpCurrentKF的Sim3，固定不动。
        CorrectedSim3[mpCurrentKF] = mg2oScw;
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();

        {

            // 地图线程。
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // 步骤2.1 通过位姿传递，得到Sim3调整与当前关键帧相连的其他关键帧的位姿（没有修正）。
            for(vector<KeyFrame *>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
            {
                KeyFrame *pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();

                // currentKF在上面添加过。
                if(pKFi != mpCurrentKF)
                {
                    // 得到关键帧pKFi的相对变换。
                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                    cv::Mat tic = Tic.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);

                    // 当前关键帧的位姿固定，其他相连关键帧根据先对关系得到Sim3调整后的位姿。
                    g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;

                    // 得到Sim3闭环g2o优化后与当前关键帧相连的各帧位姿。
                    CorrectedSim3[pKFi] = g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                cv::Mat tiw = Tiw.rowRange(0,3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);

                // 与当前关键帧相连的各关键帧，未经过Sim3闭环g2o优化的位姿。
                NonCorrectedSim3[pKFi] = g2oSiw;
            }

            // 步骤2.2 得到调整后与当前关键帧相连的各帧位姿后，修正这些关键帧的MapPoints。
            for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
            {
                KeyFrame *pKFi = mit->first;
                g2o::Sim3 g2oCorrectedSiw = mit->second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

                vector<MapPoint *> vpMPsi =  pKFi->GetMapPointMatches();

                // 遍历关键帧的点云。
                for(size_t iMP=0, endMPi=vpMPsi.size(); iMP<endMPi; iMP++)
                {
                    MapPoint *pMPi = vpMPsi[iMP];
                    if(!pMPi)
                        continue;
                    if(pMPi->isBad())
                        continue;
                    // 当前关键帧点云保持不动。
                    if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    // 将未校正的P3Dw(点云)从世界坐标映射到未校正的pKFi相机坐标系，再映射到校正后的世界坐标。
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                    // eigP3Dw 世界坐标 -> 未校正相机坐标 -> 校正后世界坐标。
                    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateNormalAndDepth();
                }

                // 步骤2.3 将sim3转换为SE3,根据Sim3更新关键帧位姿。
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                // R t/s
                // 0  1
                eigt *=(1.0/s);
                cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);
                pKFi->SetPose(correctedTiw);

                // 根据共视关系更新当前关键帧与其帧之间的连接。
                pKFi->UpdateConnections();
            }

            // 步骤3 检查当前帧的MapPoints与闭环匹配帧的MapPoints（当前关键帧与闭环帧的匹配）是否有冲突，对冲突的MapPoints进行替换或填补。
            for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
            {
                if(mvpCurrentMatchedPoints[i])
                {
                    MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
                    MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                    // 如果有重复的MapPoint，则用匹配帧的代替。
                    if(pCurMP)
                        pCurMP->Replace(pLoopMP);
                    // 如果没有该MapPoint，直接添加。
                    else
                    {
                        mpCurrentKF->AddMapPoint(pLoopMP, i);
                        pLoopMP->AddObservation(mpCurrentKF, i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }

        }


        // 步骤4 通过将与闭环帧相连的关键帧的点云mvpLoopMapPoints投影到关键帧中，进行MapPoint检查与替换。
        SearchAndFuse(CorrectedSim3);

        // 步骤5 更新当前关键的共视关系，闭环时的MapPoints融合会得到新的连接。
        map<KeyFrame *, set<KeyFrame *> > LoopConnections;

        // 步骤5.1 遍历与当前关键帧相邻的关键帧(1级)。
        // mvpCurrentConnectedKFs是sim3校正更新前的连接关系。
        for(vector<KeyFrame *>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame *pKFi = *vit;
            
            // 步骤5.2 得到与当前关键帧相连的关键帧的相连帧(2级)。
            vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // 步骤5.3 更新1级相连关键帧的连接关系。
            pKFi->UpdateConnections();

            // 步骤5.4 取出该帧更新后的连接关系。
            LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();

            // 获取由闭环得到的连接关系，用于Essential优化。
            // 步骤5.5 从连接关系中去除闭环之前的2级连接。
            for(vector<KeyFrame *>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev);
            }
            // 步骤5.6 从连接关系中去除闭环之前的1级连接，剩下的是闭环后得到的连接关系。
            for(vector<KeyFrame *>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // 步骤6 进行Essential图优化，LoopConnections是形成闭环后新生成的连接关系。
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

        // 步骤7 添加当前帧与闭环匹配帧之间的连接关系，这个连接关系不优化。
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // 步骤8 新建一个线程，用于全局BA。
        // Essential优化只优化了关键帧的位姿，这里全局优化所有位姿和点云。
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

        // 闭环完成，继续局部地图。
        mpLocalMapper->Release();

        cout << "Loop Closed!" << endl;

        // 记录闭环是当前关键帧Id。
        mLastLoopKFid = mpCurrentKF->mnId;

    }



    // 通过将闭环时相连关键帧的MapPoints投影到这些关键帧中，进行MapPoints的检查和替换。
    void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        ORBmatcher matcher(0.8);

        // 遍历所有闭环后与当前关键帧相连的关键帧。
        for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            g2o::Sim3 g2oScw = mit->second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            // 将闭环后与当前关键帧相连的关键帧的MapPoint变换到当前关键帧下坐标系，投影，检查冲突并融合。
            // mvpLoopMapPoints记录需要替换的点，调用Replace进行替换。
            vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint *>(NULL) );
            matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

            // 获取Map线程。
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for(int i=0; i<nLP; i++)
            {
                MapPoint *pRep = vpReplacePoints[i];
                if(pRep)
                {
                    // 用闭环点云mvpLoopMapPoints替换之前的点云。
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }

    }



    // 重置请求，设置重置标志位。
    void LoopClosing::RequestReset()
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

            // 延时5000us。
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        }

    }

    // 检测重置申请标志位后，重置操作。
    void LoopClosing::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            mlpLoopKeyFrameQueue.clear();
            mLastLoopKFid = 0;
            mbResetRequested = false;
        }

    }

    // 运行全局BA，在单独的线程中。
    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        cout << "Starting Global Bundle Adjustment " << endl;

        Optimizer::GlobalBundleAdjustment(mpMap, 20, &mbStopGBA, nLoopKF, false);
        
        // 更新所有点云和关键帧。

        // 在全局BA期间， 局部地图线程正常工作，可能会插入不包括在全局BA中的新关键帧。
        // 新插入关键帧与更新的地图是不一致的，需要通过Spanning tree校正。
        {
            unique_lock<mutex> lock(mMutexGBA);

            // GBA没有运行完成，没有被终止。
            if(!mbStopGBA)
            {
                cout << "Global Bundle Adjustment Finished." << endl;
                cout << "Updating map ..." << endl;
                
                // 发送申请，停止Local Mapping。
                mpLocalMapper->RequestStop();

                // 等待，直到Local Mapping线程完成。
                while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                // 获取地图线程互斥变量。
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // 从地图的第一帧开始校正关键帧。
                list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

                while(!lpKFtoCheck.empty())
                {
                    // 地图中的关键帧。
                    KeyFrame *pKF = lpKFtoCheck.front();
                    const set<KeyFrame *> sChilds = pKF->GetChilds();
                    cv::Mat Twc = pKF->GetPoseInverse();

                    for(set<KeyFrame *>::const_iterator sit=sChilds.begin(); sit!=sChilds.end(); sit++)
                    {
                        KeyFrame *pChild = *sit;

                        // 关键帧pKF的子关键帧的全局BA优化关键帧不是闭环时的当前关键帧。
                        if(pChild->mnBAGlobalForKF != nLoopKF)
                        {
                            // pKF到子关键帧的位姿变换。
                            cv::Mat Tchildc = pChild->GetPose()*Twc;
                            // 校正后世界坐标系到子关键帧的位姿变换。
                            pChild->mTcwGBA = Tchildc*pKF->mTcwGBA; 
                            // 避免重复添加。
                            pChild->mnBAGlobalForKF = nLoopKF;
                        }

                        // 关键帧pKF的子关键帧加入地图关键帧队列。
                        lpKFtoCheck.push_back(pChild);
                    }

                    // 保存GBA之前的位姿。
                    pKF->mTcwBefGBA = pKF->GetPose();
                    // 更新用GBA之后的位姿。
                    pKF->SetPose(pKF->mTcwGBA);
                    lpKFtoCheck.pop_front();
                }

                // 校正地图点云。
                const vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

                for(size_t i=0; i<vpMPs.size(); i++)
                {
                    MapPoint *pMP = vpMPs[i];

                    if(pMP->isBad())
                        continue;

                    // 运行GBA时的关键帧是闭环时的当前帧。
                    if(pMP->mnBAGlobalForKF==nLoopKF)
                    {
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // 利用它的参考关键帧校正更新点云。
                        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                        if(pRefKF->mnBAGlobalForKF != nLoopKF)
                            continue;

                        // 利用未校正参考帧位姿，获得pMP在参考帧坐标系下的坐标。
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                        cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                        // 利用校正后的参考帧位姿，获得pMP在世界坐标系下的校正位姿。
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                        cv::Mat twc = Twc.rowRange(0,3).col(3);
                        pMP->SetWorldPos(Rwc*Xc+twc);
                    }
                }   // 点云地图校正。

                mpLocalMapper->Release();
                cout << "Map updated!" <<endl;
            }   // if(!mbStopGBA)。

            // 设置全BA状态标志位。
                mbFinishedGBA = true;
                mbRunningGBA = false;
        }

    }



    // 设置线程完成请求标志位。
    void LoopClosing::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    // 检查完成标志位。
    bool LoopClosing::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    // 设置线程完成标志位,表示线程结束。
    void LoopClosing::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    // 线程是否结束。
    bool LoopClosing::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }



}   // namespace ORB_SLAM2














#include "Optimizer.h"
#include "Converter.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/batch_stats.h"

#include <Eigen/StdVector>
#include <mutex>



namespace ORB_SLAM2
{

    // pMP中所有的MapPoints和关键帧进行全局BA优化。
    // 该GBA在ORB中的两个地方使用。
    // 1. 单目初始化，CreateInitialMapMonocular函数。
    // 2. 闭环完成后优化，RunGlobaBundleAdjustment函数。
    void Optimizer::GlobalBundleAdjustment(Map *pMap, int nIterations, bool  *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        // 获取当前地图的所有关键帧和点云。
        vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
        BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);

    }



    /**
    * 3D-2D 最小化重投影误差，e =(u,v)-project(Tcw*Pw)
    * 1.  Vertex(顶点，优化变量)
    *       g2o::VertexSE3Expmap;       // 当前帧的Tcw。
    *       g2o::VertexSBAPointXYZ;     // 当前帧的世界坐标系下的地图点云。
    * 2.  Edge(边，目标函数)
    *       g2o::EdgeSE3ProjectXYZ, BaseBinaryEdge。
    *           连接顶点：待优化当前帧Tcw。
    *                 待优化世界坐标系下的MapPoint。
    *           测量值：  MapPoint在当前帧中对应的图像坐标(u,v)。
    *           信息矩阵：invSigma2(与特征点所在的尺度有关)。
    *
    * Param
    *       vpKFs               关键帧
    *       vpMP                MapPoints
    *       nIterations         迭代次数(20次)
    *       pbStopFlag          是否强制暂停
    *       nLoopKF             关键帧的个数
    *       bRobust             是否使用核函数(代替2范数)
    */
    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                    int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {

        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        // 步骤1 初始化g2o优化器。
        // 构造求解器。
        g2o::SparseOptimizer optimizer;

        // 选择线性求解器的方法。
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        // 6×3参数
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        // L-M下降。
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        // 步骤2 向优化器添加顶点。

        // 步骤2.1 向优化器添加关键帧位姿顶点。
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if(pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));                // 设置估计量，即迭代初值。
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId==0);
            optimizer.addVertex(vSE3);
            if(pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);

        // 步骤2.2 向优化器添加MapPoint顶点。
        for(size_t i=0; i<vpMP.size(); i++)
        {
            MapPoint *pMP = vpMP[i];
            if(pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));         // 设置估计量，即迭代初值。
            const int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            // 获取观测该点云的关键帧和点云在KF中的索引。
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();    

            int nEdges = 0;
            // 步骤3 向优化器添加投影边，即待优化的目标函数。
            // 遍历可以观测到该点云的关键帧，并查找对应的2D图像坐标。
            for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
            {
                KeyFrame *pKF = mit->first;
                if(pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
                
                // 单目或RGB-D。
                if(pKF->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    // 设置边连接的顶点。
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(id)));    
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);                                         // 设置观测值，即测量值，误差函数中的(u,v)。
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];    // 提取特征对应图像金字塔的尺度。
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);       // 信息矩阵。

                    // 是否使用核函数。
                    if(bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                }

                // 双目。
                else
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    if(bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
					e->bf = pKF->mbf;

                    optimizer.addEdge(e);
                }
            }

            if(nEdges == 0)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            }
            else
            {
                vbNotIncludedMP[i] = false;
            }

        }

        // 步骤4 开始优化。
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

        // 步骤5 得到优化结果。

        // 关键帧。
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if(pKF->isBad())
                continue;

            // 保存优化后结果。
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat= vSE3->estimate();

            if(nLoopKF == 0)
            {
                pKF->SetPose(Converter::toCvMat(SE3quat));
            }
            else
            {
                pKF->mTcwGBA.create(4, 4, CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        // 地图点云。
        for(size_t i=0; i<vpMP.size(); i++)
        {
            if(vbNotIncludedMP[i])
                continue;

            MapPoint *pMP = vpMP[i];

            if(pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId+maxKFid+1));

            if(nLoopKF == 0)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }

    }



    /*
    * pose 图优化
    *   3D-2D最小化重投影误差 e = (u,v) - project(Tcw*Pw)
    *
    * Vertex(顶点，优化变量)
    *   g2o::VertexSE3Expmap()          // 当前帧的Tcw。
    * Edge(边，目标函数)
    *   g2o::EdgeSE3ProjectXYZOnlyPose, BaseUnaryEdge。
    *       连接顶点：待优化当前帧位姿Tcw。
    *       测量值：当前地图点云MP在当前帧尺度下的图像坐标(u,v)。
    *       信息矩阵：invSigma2(与特征点尺度有关)。
    *   g2o::EdgeStereoSE3ProjectXYZOnlyPose, BaseUnaryEdge。
    *       连接顶点：待优化当前帧位姿Tcw。
    *       测量值： MapPoint在当前帧中的图像坐标(ul,vl, ur)。
    *       信息矩阵：invSigma2(与特征点尺度有关)。
    *
    * Param     pFrame Frame
    * return    inliers数量。
    */
    // 该优化函数主要用于Tracking线程中，运动跟踪，参考帧跟踪，地图跟踪，重定位。
    int Optimizer::PoseOptimization(Frame *pFrame)
    {
        
        // 步骤1 构造g2o优化器。
        g2o::SparseOptimizer optimizer;         // 构造求解器

        // 选择线性模型的求解方法。
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        // 6×3参数设置。
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // 步骤2 添加顶点：当前待优化变量Tcw。
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));      // 待优化变量的初始值。
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Frame特征点数量。
        const int N = pFrame->N;
        
        // 单目情况。
        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;        // 存储一元边。
        vector<size_t> vnIndexEdgeMono;                             // 存储内点的索引号。
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        // 双目情况。
        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        // 步骤3 添加一元边：
        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if(pMP)
                {

                    // 单目。
                    if(pFrame->mvuRight[i] < 0)
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));     // 设置顶点。
                        e->setMeasurement(obs);                                                                 // 设置测量值(u,v)。
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        // 使用Huber核函数。
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }

                    // 双目
                    else
                    {

                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }
            }
        }

        // 初始特征点配对数量少于3，结束。
        if(nInitialCorrespondences < 3)
            return 0;


        // 步骤4 优化计算。
        // 总共进行4次优化， 每次优化后，将观测点分为outlier和inlier，outlier不参与下次优化。
        // 每次优化后重新判断所有观测点，之前的outlier可能变为inlier，反之亦然。
        // 基于卡方分布计算阈值(假设测量有一个像素偏差)。
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};                            // 迭代次数。

        int nBad = 0;
        for(size_t it=0; it<4; it++)
        {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));      // 优化4次，每次优化设置初值。
            optimizer.initializeOptimization(0);                        // 对level为0进行优化。
            optimizer.optimize(its[it]);                                // 运行优化。

            nBad = 0;

            // 遍历所有的边。
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];
                const size_t idx = vnIndexEdgeMono[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();              // 只计算active edge的误差。
                }
                
                const float  chi2 = e->chi2();
                
                if(chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);                 // 设置为outlier。
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);                 // 设置为inlier。
                }

                if(it == 2)
                    e->setRobustKernel(0);         // 只有前两次优化使用核函数。
            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];
                const size_t idx = vnIndexEdgeStereo[i];

                if(pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2 > chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if(it == 2)
                    e->setRobustKernel(0);
            }

            if(optimizer.edges().size() < 10)
                break;
        }

        // 步骤5 保存优化后位姿，返回内点数量。
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences-nBad;
    }



    /*
    * 局部BA优化
    *   顶点：
    *       g2o::VertexSE3Expmap,   LocalKeyFrames，即当前关键帧，与当前关键帧相连关键帧的位姿。
    *       g2o::VertexSE3Expmap,   FixedCameras,   即能观测到Local MapPoints 且不属于LocalKeyFrames的关键帧的位姿，在优化中，这些关键帧位姿不变。
    *       g2o::VertexSBAPointXYZ, LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的坐标。
    *
    *   边：
    *       g2o::EdgeSE3ProjectXYZ, BaseBinaryEdge
    *           连接顶点：关键帧的位姿Tcw, MapPoint的世界坐标系坐标Pw。
    *           测量值：  MapPoint在关键帧中的图像坐标。
    *           信息矩阵：与特征点尺度有关。
    *   Param:
    *       PKF                 KeyFrame
    *       pbStopFlag          是否停止优化的标志。
    *       pMap                在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate。
    *
    */
    // 该函数用于LocalMapping线程的局部BA优化。
    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
    {

        list<KeyFrame *> lLocalKeyFrames;

        // 步骤1 将当前帧加入到lLocalKeyFrames。
        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        // 步骤2 找到当前关键帧pKF连接的关键帧(一级相连)，加入lLocalKeyFrames。
        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
        {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        // 步骤3 遍历所有lLocalKeyFrames中的关键帧，将观测到的MapPoints加入到lLocalMapPoints。
        list<MapPoint *> lLocalMapPoints;
        for(list<KeyFrame *>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();    // 获取该帧所有的MapPoints。
            for(vector<MapPoint *>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint *pMP = *vit;
                if(pMP)
                {
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;        // 防止冲突添加不同关键帧观测到的相同点云。
                        }
                }
            }
        }

        // 步骤4 获得能观测到局部MapPoints，但不属于局部关键帧的关键帧，这些关键帧在Local中不优化。
        list<KeyFrame *> lFixedCameras;
        for(list<MapPoint *>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame *, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                // pKFi->mnBALocalForKF!=pKF->mnId表示不是局部关键帧。
                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;       // 防止重复添加。
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // 步骤5 构造g2o优化器。
        g2o::SparseOptimizer optimizer;         // g2o求解器。
        
        // 设置线性模型的求解算法。
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        // 6×3性质。
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        
        // L-M下降。
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // 步骤6 添加顶点，局部关键帧位姿。
        for(list<KeyFrame *>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));                           // 设置迭代初值。
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==0);                                                      // 第一帧位置固定。
            optimizer.addVertex(vSE3);
            if(pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // 步骤7 添加顶点， 非LocalKeyFrames关键帧，固定不优化。
        for(list<KeyFrame *>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));                           // 设置迭代初值。
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // 步骤8 添加顶点， LocalMapPoints, 3D点坐标。
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        // 遍历所有Local MapPoint。
        for(list<MapPoint *>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            // 添加顶点，局部MapPoints。
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));          // 设置迭代初值。
            int id = pMP->mnId+maxKFid+1;                                           // 防止id和LocalKF冲突。
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            // 步骤9 对每一对关联的MapPoint和KF构建边。
            for(map<KeyFrame *, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if(!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // 单目。
                    if(pKFi->mvuRight[mit->second] < 0)
                    {

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;            // 测量值(u,v)。

                        g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                        // 设置边连接的顶点。
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }

                    // 双目
                    else
                    {
                        Eigen::Matrix<double, 3, 1> obs;
                        const float kp_ur = pKFi->mvuRight[mit->second];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
						e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }
                }
            }
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        // 步骤10 开始优化。
        optimizer.initializeOptimization();
        optimizer.optimize(5);                 // 执行优化。

        bool bDoMore = true;

        // 判断是否要求中断BA。
        if(pbStopFlag)
            if(*pbStopFlag)
                bDoMore = false;

        if(bDoMore)
        {
            // 步骤11 检测outlier，并设置下次不优化。
            // 单目
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if(pMP->isBad())
                    continue;

                // 基于卡方检验计算出的阈值。
                if(e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);         // 不优化。
                }

                e->setRobustKernel(0);     // 下次不使用核函数。
            }

            // 双目
            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                MapPoint *pMP = vpMapPointEdgeStereo[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // 步骤12 剔除outlier后再次优化。
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

        // 步骤13 重新优化后，剔除误差比较的边连接的KF和MapPoint。
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];
            
            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        // 获取Map线程锁。
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // 步骤14 剔除投影误差过大的关键帧和地图点，在关键帧中剔除该对MapPoints的观测， 在该MapPoint中剔除该关键帧对其的观测。
        if(!vToErase.empty())
        {
            for(size_t i=0; i<vToErase.size(); i++)
            {
                KeyFrame *pKFi = vToErase[i].first;
                MapPoint *pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // 步骤15 优化更新后的关键帧位姿以及MapPoints的位置，平均观测方向等属性。

        // 关键帧属性更新。
        for(list<KeyFrame *>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame *pKF = *lit;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        // MapPoint属性更新。
        for(list<MapPoint *>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

    }



    /*
    * 闭环检测后，进行Essential图优化。
    *
    *   顶点：
    *       g2o::VertexSim3Expmap, Essential图中的关键帧的位姿。
    *   边：
    *       g2o::EdgeSim3(), BaseBinaryEdge。
    *           连接的顶点：关键帧位姿Tcw, MapPoint Pw。
    *           测量值：    经过CorrectLoop函数步骤2后，sim3矫正后的位姿。
    *           信息矩阵：  单位阵。
    *
    *   Param
    *       pMap                    全局地图。
    *       pLoopKF                 闭环匹配上的关键帧。
    *       pCurKF                  当前关键帧。
    *       NonCorrectedSim3        未经过sim3调整过的关键帧位姿。
    *       CorrectedSim3           经过sim3调整过的关键帧位姿。
    *       LoopConnections         因闭环时MapPoints调整重新产生的边。
    *
    **/
    void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF, 
                                            const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                            const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                            const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
    {

        // 步骤1 构造优化器。
        g2o::SparseOptimizer optimizer;             // 构造求解器。
        optimizer.setVerbose(false);                // 调试信息不输出。

        // 选择Eigen块求解器作为线性模型的求解方法。
		g2o::BlockSolver_7_3::LinearSolverType *linearSolver = 
        		new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();

        // 构造线性求解器。
        g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);

        // 使用L-M算法进行非线性迭代。
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        solver->setUserLambdaInit(1e-16);       // L-M中的lambda。
        optimizer.setAlgorithm(solver);

        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        // 经过Sim3调整，未优化的KF位姿。
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
        // 经过Sim3调整，优化后的KF位姿。
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid+1);

        const int minFeat = 100;

        // 步骤2 添加顶点：地图中所有KF的pose。
        // 尽量使用经过sim3调整的位姿。
        for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if(pKF->isBad())
                continue;

            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

            const int nIDi = pKF->mnId;

            LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

            // 如果关键帧在闭环时通过sim3调整过位姿，使用调整位姿。
            if(it!=CorrectedSim3.end())
            {
                vScw[nIDi] = it->second;
                VSim3->setEstimate(it->second);             // 设置迭代初始值。
            }

            // 没有经过sim3调整，使用自身位姿。
            else
            {
                Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw, tcw, 1.0);
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
            }

            // 闭环匹配到的帧不进行位姿优化
            if(pKF==pLoopKF)
                VSim3->setFixed(true);

            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);
            VSim3->_fix_scale = bFixScale;

            optimizer.addVertex(VSim3);

            // 优化前的pose顶点。
            vpVertices[nIDi] = VSim3;
        }
        
        set<pair<long unsigned int, long unsigned int> > sInsertedEdges;
        const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

        // 步骤3 添加边：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系，不是当前帧与闭环匹配帧之间的连接关系。
        for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit=LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            const long unsigned int nIDi = pKF->mnId;
            const set<KeyFrame *> &spConnections = mit->second;
            const g2o::Sim3 Siw = vScw[nIDi];
            const g2o::Sim3 Swi = Siw.inverse();

            for(set<KeyFrame *>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
            {
                const long unsigned int nIDj = (*sit)->mnId;
                if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                    continue;

                const g2o::Sim3 Sjw = vScw[nIDj];
                // 得到两个pose之间的sim3变换。
                const g2o::Sim3 Sji = Sjw*Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(nIDj)));    // 添加边的顶点，sp中的帧。
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(nIDi)));    // 添加边的顶点，sp树的父节点帧。
                e->setMeasurement(Sji);                                                                     // 边的测量值，两顶点间的相对位姿变换。

                e->information() = matLambda;                   // 单位阵。

                optimizer.addEdge(e);                           // 添加边。

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        // 步骤4 添加跟踪时形成的边，闭环匹配成功形成的边。
        for(size_t i=0, iend=vpKFs.size(); i<iend; i++)            // 遍历地图中的所有关键帧。
        {
            KeyFrame *pKF = vpKFs[i];

            const int nIDi = pKF->mnId;

            g2o::Sim3 Swi;

            LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

            // 尽可能得到没有经过sim3调整的边。
            if(iti!=NonCorrectedSim3.end())
                Swi = (iti->second).inverse();
            else
                Swi = vScw[nIDi].inverse();

            KeyFrame *pParentKF = pKF->GetParent();

            // 步骤4.1 只添加扩展树的边，有父节点的关键帧。
            if(pParentKF)
            {
                int nIDj = pParentKF->mnId;

                g2o::Sim3 Sjw;

                LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                // 尽可能得到未经过Sim3传播调整的位姿。
                if(itj!=NonCorrectedSim3.end())
                    Sjw = itj->second;
                else
                    Sjw = vScw[nIDj];

                g2o::Sim3 Sji = Sjw*Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(nIDj)));    // 地图中某一关键帧的父节点。
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(nIDi)));    // 地图中某一关键帧。
                e->setMeasurement(Sji);             // 测量值，边两个顶点的位姿相对变换。

                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // 步骤4.2 添加在CorrectLoop函数中AddLoopEdge函数添加的连接边(当前关键帧与闭环帧的连接关系)。
            const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();         // 获取闭环帧。

            for(set<KeyFrame *>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
            {
                KeyFrame *pLKF = *sit;
                if(pLKF->mnId < pKF->mnId)
                {
                    g2o::Sim3 Slw;
                    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                    // 尽量使用未经过sim3调整的位姿。
                    if(itl!=NonCorrectedSim3.end())
                        Slw = itl->second;
                    else
                        Slw = vScw[pLKF->mnId];

                    g2o::Sim3 Sli = Slw*Swi;
                    g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(pLKF->mnId)));     // 地图中的某帧关键帧的闭环帧。
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *> (optimizer.vertex(nIDi)));           // 地图中的某帧关键帧。
                    el->setMeasurement(Sli);        // 边的两顶点的相对位姿变换。
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                }
            }

            // 步骤4.3 与pKF具有很好共视关系的关键帧也作为边进行优化。
            const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for(vector<KeyFrame *>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
            {
                
                KeyFrame *pKFn = *vit;
                if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                {
                    if(!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                    {
                        if(sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId)))) // 在步骤3中已经添加过。
                            continue;

                        g2o::Sim3 Snw;
                        LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                        // 尽量使用未经过sim3调整的位姿。
                        if(itn != NonCorrectedSim3.end())
                            Snw = itn->second;
                        else
                            Snw = vScw[pKFn->mnId];

                        g2o::Sim3 Sni = Snw*Swi;         // 相对位姿变换，地图中某帧关键帧和pKFn的位姿相对变换。

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));  // 添加边，pKFn。
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));        // 添加边，pKF。
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                    }
                }
            }
        }

        // 步骤5 开始优化。
        optimizer.initializeOptimization();
        optimizer.optimize(20);

		 unique_lock<mutex> lock(pMap->mMutexMapUpdate);
		 
        // 步骤6 设置优化后的位姿。Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame *pKFi = vpKFs[i];
            const int nIDi = pKFi->mnId;

            g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
            g2o::Sim3 CorrectedSiw = VSim3->estimate();
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
            Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = CorrectedSiw.translation();
            double s = CorrectedSiw.scale();

            eigt *= (1.0/s);

            cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // 步骤7 步骤5和步骤6优化得到关键帧位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置。
        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint *pMP = vpMPs[i];

            if(pMP->isBad())
                continue;

            int nIDr;

            // 该MapPoint经过Sim3调整过(CorrectLoop函数的步骤2.2)。
            if(pMP->mnCorrectedByKF == pCurKF->mnId)
            {
                nIDr = pMP->mnCorrectedReference;
            }
            else
            {
                // 通常情况下，MapPoint的参考帧就是它的创建关键帧。
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            // 得到MapPoint参考关键帧步骤5优化前的位姿。
            g2o::Sim3 Srw = vScw[nIDr];
            // 得到MapPoint参考关键帧后的位姿。
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));  // 3D坐标 从w->r, r->w，假设3D点在相机坐标下的坐标不变。

            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
            pMP->SetWorldPos(cvCorrectedP3Dw);

            pMP->UpdateNormalAndDepth();
        }
    }



    /*
    * 形成闭环后进行Sim3优化，优化pKF1和pKF2之间的sim3位姿变换。
    *   顶点：
    *       g2o::VertexSim3Expmap, 两个关键帧位姿变换。
    *       g2o::VertexSBAPointXYZ, 两关键帧的MapPoints。
    *   边：
    *       g2o::EdgeSim3ProjectXYZ, BaseBinaryEdge
    *           连接顶点：关键帧的sim3位姿，地图点云Pw。
    *           测量值： MapPoint在关键帧中的图像坐标(u,v)。
    *           信息矩阵：invSigma2(与特征点尺度有关)。
    *       g2o::EdgeInverseSim3ProjectXYZ, BaseBinaryEdge
    *           连接顶点: 关键帧的sim3位姿，地图点云Pw。
    *           测量值：MapPoint在关键帧中的图像坐标(u,v)。
    *           信息矩阵：invSigma2(与特征点尺度有关)。
    *
    *   Param
    *       pKF1, pKF2          KeyFrame。
    *       vpMatches1          两关键帧之间的匹配关系。
    *       g2oS12              两关键帧之间的Sim3变换。
    *       th2                 核函数阈值。
    *       bFixedScale         是否优化尺度，单目优化，双目不优化。
    *
    *   return                  返回内点数量。
    */
    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
        // 步骤1 初始化g2o求解器。
        g2o::SparseOptimizer optimizer;                             // 构造求解器。

        // 构造线性方程求解器, Hx = -b。
        g2o::BlockSolverX::LinearSolverType *linearSolver;          
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();     // 使用Dense算法求解器。
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

        // 使用L-M迭代算法。
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // 相机内参。
        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        // 相机位姿。
        const cv::Mat R1w = pKF1->GetRotation();
        const cv::Mat t1w = pKF1->GetTranslation();
        const cv::Mat R2w = pKF2->GetRotation();
        const cv::Mat t2w = pKF2->GetTranslation();

        // 步骤2 添加顶点(优化变量)。
        
        // 步骤2.1 添加Sim3顶点，两关键帧之间位姿变换。
        g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
        vSim3->_fix_scale = bFixScale;
        vSim3->setEstimate(g2oS12);             // 设置待优化变量初值。
        vSim3->setId(0);
        vSim3->setFixed(false);                 // 加入优化。
        vSim3->_principle_point1[0] = K1.at<float>(0,2);            // cx
        vSim3->_principle_point1[1] = K1.at<float>(1,2);            // cy
        vSim3->_focal_length1[0] = K1.at<float>(0,0);               // fx
        vSim3->_focal_length1[1] = K1.at<float>(1,1);               // fy
        vSim3->_principle_point2[0] = K2.at<float>(0,2);
        vSim3->_principle_point2[1] = K2.at<float>(1,2);
        vSim3->_focal_length2[0] = K2.at<float>(0,0);
        vSim3->_focal_length2[1] = K2.at<float>(1,1);
        optimizer.addVertex(vSim3);


        const int N = vpMatches1.size();                // 匹配点数量。
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();

        vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;            // pKF2对应的MapPoints到pKF1的投影。
        vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;     // pKF1对应的MapPoints到pKF2的投影。
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2*N);
        vpEdges12.reserve(2*N);
        vpEdges21.reserve(2*N);

        const float deltaHuber = sqrt(th2);

        int nCorrespondences = 0;

        for(int i=0; i<N; i++)
        {
            if(!vpMatches1[i])
                continue;

            // pMP1和pMP2是匹配的MapPoints。
            MapPoint *pMP1 = vpMapPoints1[i];
            MapPoint *pMP2 = vpMatches1[i];

            // 边id。
            const int id1 = 2*i+1;  
            const int id2 = 2*(i+1);            

            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(pMP1 && pMP2)
            {
                if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
                {
                    // 步骤2.2 添加PointXYZ顶点
                    g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D1w = pMP1->GetWorldPos();
                    cv::Mat P3D1c = R1w*P3D1w + t1w;
                    vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                    vPoint1->setId(id1);
                    vPoint1->setFixed(true);        // 不参与优化。
                    optimizer.addVertex(vPoint1);

                    g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D2w = pMP2->GetWorldPos();
                    cv::Mat P3D2c = R2w*P3D2w + t2w;
                    vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                    vPoint2->setId(id2);
                    vPoint2->setFixed(true);        // 不参与优化。
                    optimizer.addVertex(vPoint2);
                }
                else 
                    continue;
            }
            else
                continue;

            nCorrespondences++;

            Eigen::Matrix<double, 2, 1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;

            // 步骤2.3 设置边。
            // 从pKF2到pKF1的重投影误差。
            g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setMeasurement(obs1);          // pKF1的地图点云在KF1的图像坐标。
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
            e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            Eigen::Matrix<double, 2, 1> obs2;
            const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;

            // 从pKF1->pKF2的重投影误差。
            g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();
            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setMeasurement(obs2);
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
            e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            vpEdges12.push_back(e12);
            vpEdges21.push_back(e21);
            vnIndexEdge.push_back(i);
        }

        // 步骤3 g2o开始优化，迭代5次。
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // 步骤4 outliers检查，剔除误差大的边。
        int nBad = 0;
        for(size_t i=0; i<vpEdges12.size(); i++)
        {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if(!e12 || !e21)
                continue;

            // 剔除
            if(e12->chi2()>th2 || e21->chi2()>th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
                vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
                nBad++;
            }
        }

        int nMoreIterations;
        if(nBad >0)
            nMoreIterations = 10;
        else
            nMoreIterations = 5;

        // 剩余的内点。
        if(nCorrespondences-nBad < 10)
            return 0;

        // 步骤5 再次优化。
        optimizer.initializeOptimization();
        optimizer.optimize(nMoreIterations);

        int nIn = 0;
        for(size_t i=0; i<vpEdges12.size(); i++)
        {

            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if(!e12 || !e21)
                continue;

            if(e12->chi2() > th2 || e21->chi2() > th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            }
            else
                nIn++;
        }

        // 步骤6 保存优化后的结果。
        g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
        g2oS12 = vSim3_recov->estimate();

        return nIn;
    }



}       // namespace ORB_SLAM2











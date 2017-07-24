
// 通过论文Closed-form solution of absolute orientation using unit quaternions的解析解方法。
// 利用两帧图像中的特征点在相机坐标系下的坐标， 求解两相机间的相似变换sim3的 s, R, t。
// 用于为BA提供初值。(相当于是EPnP，对极几何的作用)。
// sim3只在发生闭环时计算，解决scale drift。



#include "Sim3Solver.h"
#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM2
{

    // 构造函数。
    Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
        mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
    {
        mpKF1 = pKF1;
        mpKF2 = pKF2;

        vector<MapPoint *> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

        // 两帧间的匹配特征点。
        mN1 = vpMatched12.size();

        mvpMapPoints1.reserve(mN1);
        mvpMapPoints2.reserve(mN1);
        mvpMatches12 = vpMatched12;
        mvnIndices1.reserve(mN1);
        mvX3Dc1.reserve(mN1);
        mvX3Dc2.reserve(mN1);

        cv::Mat Rcw1 = pKF1->GetRotation();
        cv::Mat tcw1 = pKF1->GetTranslation();
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat tcw2 = pKF2->GetTranslation();

        mvAllIndices.reserve(mN1);

        size_t idx = 0;
        for(int i1=0; i1<mN1; i1++)
        {
            if(vpMatched12[i1])
            {
                MapPoint *pMP1 = vpKeyFrameMP1[i1];
                MapPoint *pMP2 = vpMatched12[i1];

                if(!pMP1)
                    continue;

                if(pMP1->isBad() || pMP2->isBad())
                    continue;

                // indexKFx是匹配特征点的索引。
                int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
                int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

                // 索引不到对应匹配点。
                if(indexKF1<0 || indexKF2<0)
                    continue;

                // 两帧对应的匹配特征点。
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

                // 提取特征对应的图像金字塔的尺度。
                const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

                mvnMaxError1.push_back(9.210*sigmaSquare1);
                mvnMaxError2.push_back(9.210*sigmaSquare2);

                // mvpMapPointsx是存放匹配点云的容器。
                mvpMapPoints1.push_back(pMP1);
                mvpMapPoints2.push_back(pMP2);
                mvnIndices1.push_back(i1);

                cv::Mat X3D1w = pMP1->GetWorldPos();
                mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);

                cv::Mat X3D2w = pMP2->GetWorldPos();
                mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);

                mvAllIndices.push_back(idx);
                idx++;
            }
        }

        mK1 = pKF1->mK;
        mK2 = pKF2->mK;

        FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
        FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

        SetRansacParameters();

    }



    // 设置RANSAC迭代参数。
    void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
    {
        mRansacProb = probability;
        mRansacMinInliers = minInliers;
        mRansacMaxIts = maxIterations;

        N = mvpMapPoints1.size();           // 对应匹配特征的数量。

        mvbInliersi.resize(N);

        // 根据对应点数量调整参数。
        float epsilon = (float)mRansacMinInliers/N;

        // 根据epsilon，probability, maxIterations设置RANSAC迭代。
        int nIterations;

        if(mRansacMinInliers == N)
            nIterations = 1;
        else 
            nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon, 3)));

        mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

        mnIterations = 0;

    }



    // RANSAC求解mvX3Dc1和mvX3Dc2之间的sim3，函数返回mvX3Dc2到mvX3Dc1之间的sim3变换。 
    cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
    {
        bNoMore = false;
        vbInliers = vector<bool>(mN1, false);
        nInliers = 0;

        // 帧间特征匹配对数少于RANSAC要求的最少内点数。
        if(N < mRansacMinInliers)
        {
            bNoMore = true;
            return cv::Mat();
        }

        vector<size_t> vAvailableIndices;

        cv::Mat P3Dc1i(3, 3, CV_32F);
        cv::Mat P3Dc2i(3, 3, CV_32F);

        int nCurrentIterations = 0;
        while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations )
        {

            nCurrentIterations++;               // 当前while迭代次数。
            mnIterations++;                     // 总迭代次数，默认最大300。

            vAvailableIndices = mvAllIndices;    // 可用匹配点索引。

            // 步骤1 任意取3组点算sim矩阵。
            for(short i=0; i<3; i++)
            {
                // 随机选取的特征点编号。
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

                // 选取的特征点索引号。
                int idx = vAvailableIndices[randi];

                // P3Dc1i P3Dc2i的坐标顺序。
                // x1 x2 x3 ...
                // y1 y2 y3 ...
                // z1 z2 z3 ...
                mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
                mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

                // 防止重复选择特征点。 
                vAvailableIndices[idx] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }

            // 步骤2 根据两组匹配的3D点，计算帧间sim3变换。
            ComputeSim3(P3Dc1i, P3Dc2i);

            // 步骤3 通过重投影误差，进行内点检测。
            CheckInliers();

            // 内点数量大于阈值。
            if(mnInliersi >= mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;
                mBestT12 = mT12i.clone();
                mBestRotation = mR12i.clone();
                mBestTranslation = mt12i.clone();
                mBestScale = ms12i;

                // 符合RANSAC阈值，保存数据，返回。
                if(mnInliersi > mRansacMinInliers)
                {
                    nInliers = mnInliersi;
                    for(int i=0; i<N; i++)
                        if(mvbInliersi[i])
                            vbInliers[mvnIndices1[i]] = true;

                    return mBestT12;
                }
            }
        }

        if(mnIterations >= mRansacMaxIts)
            bNoMore = true;

        return cv::Mat();

    }



    // 
    cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
    {
        bool bFlag;
        return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);

    }



    // 计算质心。
    // C为质心坐标，Pr为P去质心后的3D点。
    void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
    {

        // 对矩阵的每一行求和，P矩阵的行表示x,y,z, 每列是一个坐标。
        cv::reduce(P, C, 1, CV_REDUCE_SUM);
        // 求平均，得到质心。
        C = C/P.cols;

        for(int i=0; i<P.cols; i++)
        {
            Pr.col(i) = P.col(i) - C;
        }

    }



    // 参考 Horn 1987, Closed-form solution of absolute orientataion using unit quaternions
    // 整个算法的流程参考第四章的C节。
    // 利用论文的结论，求sim3。
    void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
    {
        
        // 步骤1 计算质心和相对坐标(以质心为原点的相对坐标)。
        cv::Mat Pr1(P1.size(), P1.type());      // 定义P1的相对坐标。
        cv::Mat Pr2(P2.size(), P2.type());      // 定义P2的相对坐标。
        cv::Mat O1(3, 1, Pr1.type());           // 定义P1的质心坐标。
        cv::Mat O2(3, 1, Pr2.type());           // 定义P2的质心坐标。

        // 计算质心坐标贺和相对坐标。
        ComputeCentroid(P1, Pr1, O1);
        ComputeCentroid(P2, Pr2, O2);

        // 步骤2 计算论文4.A中的矩阵M。
        cv::Mat M = Pr2*Pr1.t();

        // 步骤3 利用M矩阵的元素，计算论文4.A中矩阵N(对称阵)。
        //              |N11, N12, N13, N14|
        //       N  =   |N12, N22, N23, N24|
        //              |N13, N23, N33, N34|
        //              |N14, N24, N34, N44| 

        double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;
        
        cv::Mat N(4, 4, P1.type());

        N11 = M.at<float>(0,0) + M.at<float>(1,1) + M.at<float>(2,2);
        N12 = M.at<float>(1,2) - M.at<float>(2,1);
        N13 = M.at<float>(2,0) - M.at<float>(0,2);
        N14 = M.at<float>(0,1) - M.at<float>(1,0);
        N22 = M.at<float>(0,0) - M.at<float>(1,1) - M.at<float>(2,2);
        N23 = M.at<float>(0,1) + M.at<float>(1,0);
        N24 = M.at<float>(2,0) + M.at<float>(0,2);
        N33 = -M.at<float>(0,0)+ M.at<float>(1,1) - M.at<float>(2,2);
        N34 = M.at<float>(1,2) + M.at<float>(2,1);
        N44 = -M.at<float>(0,0)- M.at<float>(1,1) + M.at<float>(2,2);

        N = (cv::Mat_<float>(4,4)<< N11, N12, N13, N14,
                                    N12, N22, N23, N24,
                                    N13, N23, N33, N34,
                                    N14, N24, N34, N44);

        // 步骤4 计算矩阵N的最大特征值对应的特征向量。
        cv::Mat eval, evec;

        cv::eigen(N, eval, evec);   // 计算特征值和特征向量，特征值降序排列。

        // 最大特征值对应的特征向量是旋转的四元数表示形式。
        cv::Mat vec(1, 3, evec.type());
        (evec.row(0).colRange(1,4)).copyTo(vec);     // 提取四元数虚部(q1,q2,q3)放入vec中，此处vec表示旋转轴。

        // 提取旋转角度。
        double ang = atan2(norm(vec), evec.at<float>(0,0)); // 旋转角的一半。

        // vec表示 旋转轴 × 旋转角。
        vec = 2*ang*vec/norm(vec);

        mR12i.create(3, 3, P1.type());
        // 利用罗德里格斯公式计算旋转矩阵R12(从2到1)。
        cv::Rodrigues(vec, mR12i);

        // 步骤5 计算尺度s，论文2.D中的公式。
        cv::Mat P3 = mR12i * Pr2;                           // 计算Pr2的旋转映射，计算s的公式会用到R(r'li)。

        if(!mbFixScale)
        {
            double nom = Pr1.dot(P3);
            cv::Mat aux_P3(P3.size(), P3.type());
            aux_P3 = P3;
            cv::pow(P3, 2, aux_P3);                 // P3的每个元素平方。
            double den = 0;

            for(int i=0; i<aux_P3.rows; i++)
                for(int j=0; j<aux_P3.cols; j++)
                    den += aux_P3.at<float>(i,j);

            ms12i = nom/den;

        }

        else 
            ms12i = 1.0f;


        // 步骤6 计算平移t。
        mt12i.create(1, 3, P1.type());
        mt12i = O1 - ms12i*mR12i*O2;

        // 步骤7 计算sim3变换。
        //       |sR t|
        // T12 = |0  1|
        mT12i = cv::Mat::eye(4, 4, P1.type());
        
        cv::Mat sR = ms12i * mR12i;
        sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
        mt12i.copyTo(mT12i.rowRange(0,3).col(3));

        // T21
        mT21i = cv::Mat::eye(4, 4, P1.type());
        
        cv::Mat sRinv = (1.0/ms12i)*mR12i.t();
        sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
        
        cv::Mat tinv = -sRinv*mt12i;
        tinv.copyTo(mT21i.rowRange(0,3).col(3));

    }



    // 利用计算得到sim3进行内点检查。
    void Sim3Solver::CheckInliers()
    {
        vector<cv::Mat> vP1im2, vP2im1;
        Project(mvX3Dc2, vP2im1, mT12i, mK1);        // 把Frame2中的3D点经过sim3变换到Frame1中计算重投影坐标。
        Project(mvX3Dc1, vP1im2, mT21i, mK2);        // 把Frame1中的3D点经过sim3变换到Frame2中计算重投影坐标。

        mnInliersi = 0;

        for(size_t i=0; i<mvP1im1.size(); i++)
        {
            cv::Mat dist1 = mvP1im1[i] - vP2im1[i];
            cv::Mat dist2 = vP1im2[i] - mvP2im2[i];

            const float err1 = dist1.dot(dist1);
            const float err2 = dist2.dot(dist2);

            if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
            {
                mvbInliersi[i] = true;
                mnInliersi++;
            }
            else
                mvbInliersi[i] = false;
        }

    }



    cv::Mat Sim3Solver::GetEstimatedRotation()
    {
        return mBestRotation.clone();
    }

    cv::Mat Sim3Solver::GetEstimatedTranslation()
    {
        return mBestTranslation.clone();
    }

    float Sim3Solver::GetEstimatedScale()
    {
        return mBestScale;
    }



    // 相机投影模型，将世界坐标系下的地图点投影到像素坐标系下。
    void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
    {

        cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Tcw.rowRange(0,3).col(3);
        const float &fx = K.at<float>(0,0);
        const float &fy = K.at<float>(1,1);
        const float &cx = K.at<float>(0,2);
        const float &cy = K.at<float>(1,2);

        vP2D.clear();
        vP2D.reserve(vP3Dw.size());

        for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
        {

            // 世界坐标系坐标转换为相机坐标系坐标。
            cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;

            // 投影模型。
            const float invz = 1/(P3Dc.at<float>(2));
            const float x = P3Dc.at<float>(0)*invz;
            const float y = P3Dc.at<float>(1)*invz;

            // 像素坐标。
            
            vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
        }

    }



    // 讲相机坐标系下的坐标投影到像素坐标。 
    void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
    {
        const float &fx = K.at<float>(0,0);
        const float &fy = K.at<float>(1,1);
        const float &cx = K.at<float>(0,2);
        const float &cy = K.at<float>(1,2);

        vP2D.clear();
        vP2D.reserve(vP3Dc.size());

        for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
        {
            const float invz = 1.0/(vP3Dc[i].at<float>(2));
            const float x    = vP3Dc[i].at<float>(0)*invz;
            const float y    = vP3Dc[i].at<float>(1)*invz;

            vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
        }

    }



}   // namespace ORB_SLAM2







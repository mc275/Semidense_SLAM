//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"


namespace ORB_SLAM2
{

    class Sim3Solver
    {

        public:
            // 构造函数。
            Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const std::vector<MapPoint *> &vpMatched12, const bool bFixScale=true);

            void SetRansacParameters(double probability=0.99, int minInliers=6, int maxIterations=300);

            cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);
            cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

            // 获取估计的旋转矩阵R，位移t，尺度s
            cv::Mat GetEstimatedRotation();
            cv::Mat GetEstimatedTranslation();
            float GetEstimatedScale();

        protected:

            void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

            void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

            void CheckInliers();

            // 相机投影模型，世界坐标到图像坐标。
            void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
            // 相机坐标3D到图像坐标。
            void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);

        protected:

            // 关键帧和匹配。
            KeyFrame *mpKF1;
            KeyFrame *mpKF2;

            std::vector<cv::Mat> mvX3Dc1;
            std::vector<cv::Mat> mvX3Dc2;
            std::vector<MapPoint *> mvpMapPoints1;
            std::vector<MapPoint *> mvpMapPoints2;
            std::vector<MapPoint *> mvpMatches12;
            std::vector<size_t> mvnIndices1;
            std::vector<size_t> mvSigmaSquare1; 
            std::vector<size_t> mvSigmaSquare2; 
            std::vector<size_t> mvnMaxError1;
            std::vector<size_t> mvnMaxError2;
 
            int N;
            int mN1;

            // 当前估计量。
            cv::Mat mR12i;
            cv::Mat mt12i;
            float ms12i;
            cv::Mat mT12i;
            cv::Mat mT21i;
            std::vector<bool> mvbInliersi;
            int mnInliersi;

            // 当前RANSAC状态。
            int mnIterations;
            std::vector<bool> mvbBestInliers;
            int mnBestInliers;
            cv::Mat mBestT12;
            cv::Mat mBestRotation;
            cv::Mat mBestTranslation;
            float mBestScale;


            // 对于双目和RGB-D固定尺度为1。
            bool mbFixScale;

            // 随机选择的索引。
            std::vector<size_t> mvAllIndices;

            // 投影。
            std::vector<cv::Mat> mvP1im1;
            std::vector<cv::Mat> mvP2im2;

            // RANSAC概率。
            double mRansacProb;

            // RANSAC最少内点数。
            int mRansacMinInliers;

            // RANSAC最大迭代数。
            int mRansacMaxIts;

            // 内点/外点阈值。e=dist(P1,T_ij*Pj)^2<5.991*mSigma2。
            float mTh;
            float mSigma2;

            // 标定参数。
            cv::Mat mK1;
            cv::Mat mK2;

    };





}



#endif


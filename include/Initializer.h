//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace ORB_SLAM2
{

    // 仅用于单目初始化，双目和RGBD不需要。
    
    class Initializer
    {
            typedef pair<int,int> Match;
        
        public:
            
            // 构造函数，确定参考帧。
            // 使用参考帧来初始化，SLAM正式开始的第一帧。
            Initializer(const Frame &ReferenceFrame, float sigma=1.0, int iterations=200);

            // 并行计算单应矩阵和基本矩阵。
            // 选择一个合适的模型，尝试从运动中回复模型。
            // 用当前帧（CF）,也就是SLAM逻辑上的第二帧来初始化整个SLAM,得到最开始的两帧之间的R,t(2D-2D)，和点云。
            bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

        private:

            // 假设场景为平面的情况下，通过前两帧求取单应矩阵(Homography, CurrentFrame2到ReferenceFrame1),得到该模型评分。
            void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
            // 假设场景为非平面情况下，通过前两帧求取基本矩阵(Fundamental, CurrentFrame2到ReferenceFrame1),得到该模型评分。
            void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);


            // 计算Homography矩阵，被FindHomography函数调用。
            cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
            // 计算Fundamental矩阵，被FindFundamental函数调用。
            cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);


            // 计算Homography模型评分，被FindHomography函数调用。
            float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
            // 计算Fundamental模型评分，被FindFundamental函数调用。
            float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);


            // 分解矩阵F,找到合适的R,t。
            bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, 
                                vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
            // 分解矩阵H,找到合适的R,t。
            bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, 
                                vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);


            // 通过三角化方法，利用反射投影矩阵把特征点恢复为3D点。
            void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
           
            
            // 归一化三维空间点和帧间位移。
            void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

            // ReconstructF进行cheirality check, 进一步找到F分解后最合适的解。
            int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                        const vector<Match> &vMatches12, vector<bool> &vbInliers, const cv::Mat &K, vector<cv::Point3f> &vP3D,
                        float th2, vector<bool> &vbGood, float &parallax);

            // 分解F，根据内参矩阵，得到Essential矩阵。
            void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


            // 特征点存储。
            vector<cv::KeyPoint> mvKeys1;                   // 存储参考帧的特征点。（ReferenceFrame  Frame1）
            vector<cv::KeyPoint> mvKeys2;                   // 存储当前帧中的特征点。（CurrentFrame  Frame2）
            
            // 从参考帧到当前帧的匹配特征点。
            vector<Match> mvMatches12;                      // 是pair结构，存储了两帧中的匹配特征点对。
            vector<bool> mvbMatched1;                       // 记录ReferenceFrame中的每个点在CurrentFrame中是否有匹配。

            // 相机参数。
            cv::Mat mK;                                     // 相机内参。

            
            float mSigma, mSigma2;                          // 测量误差的标准差和方法。

            int mMaxIterations;                              // 计算基本矩阵或基本矩阵的RANSAC迭代次数。
            
            vector< vector<size_t> > mvSets;                  // 二维容器，外层容器的大小为RANSAC迭代次数，内层维每次迭代需要的点。 

    }; 


}   // namespace ORB_SLAM2



#endif


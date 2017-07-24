

#ifndef PROBABILITYMAPPING_H
#define PROBABILITYMAPPING_H

#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <Eigen/Core>
#include <mutex>
#include <opencv2/opencv.hpp>

// ProbabilityMapping.cpp展开ProbabilityMapping.h, ProbabilityMapping.h 展开KeyFrame.h
// KeyFrame.h中再展开ProbabilityMapping.h时，因为已经展开过了，所以不会展开，
// 而当KeyFrame.h定义ProBabilityMapping对象时, 会找不到对象的声明，这是因为未知数据类型的大小，无法分配空间。
// 但是可以定义ProBabilityMapping对象的指针，对象指针大小固定的。
// #include "KeyFrame.h"  

namespace ORB_SLAM2
{

    // 参数设定。
    #define covisN 7               // 进行立体匹配的关键帧个数。
    #define sigmaI 20              // 图像像素灰度的标准差。
    #define lambdaG 15             // 像素梯度阈值
    #define lambdaL 80             // 像素梯度与极线的夹角。
    #define lambdaTheta 45         // 配对的两帧图像的像素梯度的夹角（包括图像旋转）。
    #define lambdaN 3              // 像素逆深度在其他帧中对应逆深度一致假设最少个数。
    #define histo_length 30         
    #define th_high 100                 
    #define th_low 50
    #define THETA 0.23              // 像素灰度与梯度方差系数。
    #define NNRATIO 0.6

    #define NULL_DEPTH 999          // 深度为0



    class KeyFrame;
    class Map;

    class ProbabilityMapping
    {
        public:
            // 初步考虑用Map和MapPoints代替。
            struct depthHo
            {
                float depth;
                float sigma;
                Eigen::Vector3f Pw;     // 半稠密地图点的3D坐标。
                bool supported;
                depthHo():depth(0.0), sigma(0.0), Pw(0.0,0.0,0.0),supported(false){}
            };

            // 构造函数。
            ProbabilityMapping(Map *pMap);


            // 半稠密线程入口。
            void Run();

            // 加入一些常量深度，用于测试半稠密地图在Pangolin的显示。
            void TestSemiDenseViewer();

            // 半稠密实现。
            void SemiDenseLoop();

            // 立体搜索约束。
            void StereoSearchConstraints(KeyFrame *kf, float *min_depth, float *max_depth);

            // 极线匹配搜索。
            void EpipolarSearch(KeyFrame *kf1, KeyFrame *kf2, const int x, const int y, float pixel, float min_depth, float max_depth, depthHo *dh, cv::Mat F12, float &best_u, float &best_v, float th_pi);

            // 获取搜索范围。
            void GetSearchRange(float &umin, float &umax, int px, int py, float mind, float maxd, KeyFrame *kf1, KeyFrame *kf2);

            // 逆深度假设融合。
            void InverseDepthHypothesisFusion(const std::vector<depthHo> &h, depthHo &dist);

            // 帧内深度一致性检查。
            void IntraKeyFrameDepthChecking(cv::Mat &depth_map, cv::Mat &depth_sigma, const cv::Mat gradimg);

            // 帧间深度一致性检查。
            void InterKeyFrameDepthChecking(KeyFrame *currentKf);

	    // 进程完成请求
            void RequestFinish()
            {
                std::unique_lock<std::mutex> lock(mMutexFinish);
                mbFinishRequested = true;
            }

            // 进程结束检查。
            bool CheckFinish()
            {
                std::unique_lock<std::mutex> lock(mMutexFinish);
                return mbFinishRequested;
            }

            void Release();
	    
	    void SetFinish()
	    {
		std::unique_lock<std::mutex> lock(mMutexFinish);
		mbFinished = true;
	    }
	    
	    bool isFinished()
	    {
		std::unique_lock<std::mutex> lock(mMutexFinish);
		return mbFinished;
	    }

	    
	    
	    
        private:

            bool mbFinishRequested;
	    bool mbFinished;
            Map *mpMap;
            
            void ComputeInvDepthHypothesis(KeyFrame *kf, KeyFrame *kf2, float ustar, float ustar_var, float a, float b, float c, depthHo *dh, int x, int y);
            void GetPixelDepth(float uj, int px, int py, ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame *kf2, float &p);
            bool ChiTest(const float& a, const float& b, const float sigma_a, float sigma_b);
            void GetFusion(const std::vector<std::pair <float,float> > supported, float& depth, float& sigma);
            cv::Mat ComputeFundamental(ORB_SLAM2::KeyFrame *pKF1, ORB_SLAM2::KeyFrame *pKF2);
            cv::Mat GetSkewSymmetricMatrix(const cv::Mat &v);

        protected:
            std::mutex mMutexSemiDense; 
            std::mutex mMutexFinish;

    };


}



#endif


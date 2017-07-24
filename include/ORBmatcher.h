//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"


namespace ORB_SLAM2
{
    
    class ORBmatcher
    {
        public:
            
            // 构造函数
            ORBmatcher(float nnratio=0.6, bool checkOri=true);
            
            // 计算两个ORB描述子之间的汉明距
            static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

            /**
             *  @brief 通过投影局部地图点云到当前帧，跟踪局部地图的点云。用于跟踪局部地图
             *  把局部地图中点云投影到当前帧，则将当前帧中的地图点云。
             *  在SearchLocalPoints()中已经把Local MapPoint重投影(InFrustum() )到当前帧。
             *  并标记局部地图点云是否在当前帧有效区域内，mbTrackInView。
             *  在局部地图重投影后点附近区域根据汉明距匹配，最终利用论文中的方向投票机制剔除。
             *  @param      F                当前帧
             *  @param      vpMapPoints      局部地图点云
             *  @param      th               阈值
             *  @return                      成功匹配的点云数目
             **/
            int SearchByProjection( Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3 );

             /**
             *  @brief 通过投影上一帧地图点云到当前帧，跟踪上一帧的点云。用于跟踪前一帧
             *  跟踪上一帧的地图点云，增加当前帧的地图点云(MapPoint)。
             *  利用恒速模型估计Tcw，投影上一帧的点云到当前帧。
             *  在投影后点附近区域根据汉明距匹配，最终利用论文中的方向投票机制剔除。
             *  @param      CurrentFrame     当前帧
             *  @param      LastFrame        上一帧
             *  @param      th               阈值
             *  @param      bMono            是否为单目
             *  @return                      成功匹配的点云数目
             **/
            int SearchByProjection( Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono );

            // 投影关键帧的地图点云到当前帧，匹配点云。
            // 用于重定位
            int SearchByProjection( Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist );

            // 使用相似变换把vpPoints的点投影到关键帧上，进行匹配跟踪。
            // 用于闭环检测
            int SearchByProjection( KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th );

            
            /**
             *
             * 通过词典，对关键帧中的地图点云和当前帧中的ORB特征进行匹配。用于闭环检测和重定位
             * 通过计算点云和ORB特征的描述子的汉明距进行跟踪匹配。
             * 为了加速匹配过程，关键帧和当前帧的描述子划分道特定层的nodes中。
             * 对同属一个node的描述子计算汉明距。
             * 通过距离阈值，比值阈值和角度投票进行剔除误匹配。
             * @param       pKF               关键帧。
             * @param       F                 当前帧。
             * @param       vpMapPointMatches F中MapPoints对应的的匹配，NULL表示未匹配。
             * @reutrn                        成功匹配的点云数目。
            */
            int SearchByBoW( KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches );      // 重定位。
            int SearchByBoW( KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12 );     // 闭环检测。

            // 匹配初始化的地图(仅单目)。
            int SearchForInitialization( Frame &F1, Frame &F2,std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10 );

            // 匹配三角化后的新点云，检查极线约束。
            int SearchForTriangulation( KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                     std::vector< pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo );
            // 使用sim(3)在KF1和KF2之间进行匹配，[s12*R12|t12]。
            // 对于双目和RGBD来说，s12=1。
            int SearchBySim3( KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12,const cv::Mat &t12,const float th );

            // 将点云投影到关键帧上，检查是否有重复的点云。
            int Fuse( KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0 );

            // 利用sim(3)将点云投影到关键帧上，查看是否有重复点云。
            int Fuse( KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint*> &vpReplacePoint );

        public:

            static const int TH_LOW;
            static const int TH_HIGH;
            static const int HISTO_LENGTH;


        protected:

            bool CheckDistEpipolarLine ( const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

            float RadiusByViewingCos ( const float &viewCos );

            void ComputeThreeMaxima (std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3 );
            
            float mfNNratio;
            bool mbCheckOrientation;


    };

} // namespace ORB_SLAM2


#endif


// 定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"


#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <mutex>

namespace ORB_SLAM2
{

    class Tracking;
    class Viewer;


    class FrameDrawer
    {
        public:

            // 构造函数。
            FrameDrawer(Map *pMap);

            // 从最后一帧更新信息。
            void Update(Tracking *pTracker);

            // 绘制最后一帧的信息。
            cv::Mat DrawFrame();

        protected:

            void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

            // 帧信息
            cv::Mat mIm;
            int N;
            vector<cv::KeyPoint> mvCurrentKeys;
            vector<bool> mvbMap, mvbVO;
            bool mbOnlyTracking;
            int mnTracked, mnTrackedVO;
            vector<cv::KeyPoint> mvIniKeys;
            vector<int> mvIniMatches;
            int mState;


            Map *mpMap;

            std::mutex mMutex;

    };

}   // namespace ORB_SLAM2



#endif

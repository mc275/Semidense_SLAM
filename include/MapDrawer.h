// 定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>

#include <mutex>

namespace ORB_SLAM2
{
    class MapDrawer
    {
        public:
            // 构造函数。
            MapDrawer(Map *pMap, const string &strSettingPath);

            Map * mpMap;

            void DrawMapPoints();
			void DrawSemiDense();
            void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
            void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
            void SetCurrentCameraPose(const cv::Mat &Tcw);
            void SetReferenceKeyFrame(KeyFrame *pKF);
            void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

        private:

            float mKeyFrameSize;
            float mKeyFrameLineWidth;
            float mGraphLineWidth;
            float mPointSize;
            float mCameraSize;
            float mCameraLineWidth;

            cv::Mat mCameraPose;

            std::mutex mMutexCamera;

    };

}   // namespace ORB_SLAM2


#endif

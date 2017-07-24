// 定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef VIEWER_H
#define VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{
    class Tracking;
    class FrameDrawer;
    class MapDrawer;
    class System;


    class Viewer
    {
        public:
            // 构造函数。
            Viewer(System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath);

            // 主线程函数，绘制点，关键帧，当前相机的位姿和最后处理的一帧。
            // 根据相机的帧率对绘制进行更新，这里采用Pangolin。

            void Run();

            void RequestFinish();

            void RequestStop();

            bool isFinished();

            bool isStopped();

            void Release();


        private:

            bool Stop();

            System *mpSystem;
            FrameDrawer *mpFrameDrawer;
            MapDrawer *mpMapDrawer;
            Tracking *mpTracker;

            // 1/fps in ms
            double mT;
            float mImageWidth, mImageHeight;

            float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

            bool CheckFinish();
            void SetFinish();
            bool mbFinishRequested;
            bool mbFinished;
            std::mutex mMutexFinish;

            bool mbStopped;
            bool mbStopRequested;
            std::mutex mMutexStop;

    };

}   // namespace ORB_SLAM2


#endif 

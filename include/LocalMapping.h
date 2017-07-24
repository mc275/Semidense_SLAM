//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace ORB_SLAM2
{
    class Tracking;
    class LoopClosing;
    class Map;

    class LocalMapping
    {

        public:
            // 构造函数。
            LocalMapping(Map *pMap, const float bMonocular);

            void SetLoopCloser(LoopClosing *pLoopCloser);

            void SetTracker(Tracking *pTracker);

            // 主函数。
            void Run();

            // 插入关键帧。
            void InsertKeyFrame(KeyFrame *pKF);

            // 线程同步。
            void RequestStop();
            void RequestReset();
            bool Stop();
            void Release();
            bool isStopped();
            bool stopRequested();
            bool AcceptKeyFrames();
            void SetAcceptKeyFrames(bool flag);
            bool SetNotStop(bool flag);

            void InterruptBA();

            void RequestFinish();
            bool isFinished();

            int KeyframesInQueue()
            {
                unique_lock<std::mutex> lock(mMutexNewKFs);
                return mlNewKeyFrames.size();
            }

        protected:

            // 是否有新关键帧。
            bool CheckNewKeyFrames();
            // 处理新关键帧。
            void ProcessNewKeyFrame();
            // 创建地图点云。
            void CreateNewMapPoints();

            // 地图点云剔除。
            void MapPointCulling();
            // 搜索临近区域。
            void SearchInNeighbors();

            // 关键帧剔除。
            void KeyFrameCulling();

            // 计算KF1和KF2之间的基本矩阵F。
            cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
            
            // 计算反对称矩阵。
            cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

            // 单目标志符。
            bool mbMonocular;
			// 
            void ResetIfRequested();            
			// 重置标识符。
			bool mbResetRequested;
			std::mutex mMutexReset;
			
			bool CheckFinish();
			void SetFinish();
            bool mbFinishRequested;
	    bool mbFinished;
            std::mutex mMutexFinish;

            Map *mpMap;

            LoopClosing *mpLoopCloser;
            Tracking *mpTracker;

            // 跟踪线程插入的关键帧先插入到该队列。
            std::list<KeyFrame *> mlNewKeyFrames;

            KeyFrame *mpCurrentKeyFrame;

            std::list<MapPoint *> mlpRecentAddedMapPoints;

            std::mutex mMutexNewKFs;

            // 不进行全局BA。
            bool mbAbortBA;

            // 进程状态标识符。
            bool mbStopped;
            bool mbStopRequested;
            bool mbNotStop;
            std::mutex mMutexStop;

            bool mbAcceptKeyFrames;
            std::mutex mMutexAccept;

    };

}   // namespace ORB_SLAM2


#endif



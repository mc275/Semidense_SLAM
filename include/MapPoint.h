//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef MAPPOINT_H
#define MAPPOINT_H


#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <mutex>

namespace ORB_SLAM2
{

    class KeyFrame;
    class Map;
    class Frame;

    // 地图点云。

    class MapPoint
    {
        public:
            // 构造函数
            MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);
            MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);
            
            // 
            void SetWorldPos(const cv::Mat &Pos);
            cv::Mat GetWorldPos();
            
            // 
            cv::Mat GetNormal();
            KeyFrame *GetReferenceKeyFrame();

            // 
            std::map<KeyFrame *,size_t> GetObservations();
            int Observations();

            // 添加/删除观测，记录KF中可以观测到该地图点云的特征点。
            void AddObservation(KeyFrame *pKF, size_t idx);
            void EraseObservation(KeyFrame *pKF);
            
            // 获取索引号，检测点云是否在KF中。
            int GetIndexInKeyFrame(KeyFrame *pKF);
            bool IsInKeyFrame(KeyFrame *pKF);

            // 告知观测到该MapPoint的KF，该点已经被删除。
            void SetBadFlag();

            // 没有通过MapPointCulling检测的Mappoint。
            bool isBad();

            // 在形成闭环时，更新MapPoint和KF之间的关系。
            void Replace(MapPoint *pMP);
            MapPoint *GetReplaced();

            // 增加可见。
            // 可见表示 
            // 1. 该点云在某些帧的范围内。2.可以被一些帧观测到，但是没有与这些帧的特征点匹配上。
            void IncreaseVisible(int n=1);
            
            // 可以找到该点的帧数+n,默认为1。
            void IncreaseFound(int n=1);

            // 
            float GetFoundRatio();
            
            inline int GetFound()
            {
                return mnFound;
            }

            // 计算具有代表性的描述子。
            void ComputeDistinctiveDescriptors();

            // 获取描述子。
            cv::Mat GetDescriptor();

            // 更新平均观测方向和观测距离。
            void UpdateNormalAndDepth();

            // 最小，最大观测深度的倒数。
            float GetMinDistanceInvariance();
            float GetMaxDistanceInvariance();

            // 预测尺度。
            int PredictScale(const float &currentDist, const float &logScaleFactor);


        public:

            long unsigned int mnId;                 // 点云的全局Id。
            static long unsigned int nNextId;
            const long int mnFirstKFid;             // 创建点云的关键帧Id。
            const long int mnFirstFrame;            // 创建该点云的帧Id。
            int nObs;

            // 用于Tracking变量。
            float mTrackProjX;
            float mTrackProjY;
            float mTrackProjXR;
            int mnTrackScaleLevel;
            float mTrackViewCos;

            // TrackLocalMap - SearchByProjection 决定是否对该点进行投影。
            // mbTrackInView==false的点有以下几种：
            // a 已经和当前帧经过匹配(TrackReference, TrackWithMotionModel) 但在优化过程中认为时外点。
            // b 已经和当前帧经过进过匹配，内点，不需要投影。
            // c 不在当前相机事业中。（未通过isInFrustum判断）。
            bool mbTrackInView;

            // TrackLocalMap - UpdateLocalPoints中防止将MapPoint重复添加到mvpLocalMapPoints的标记。
            long unsigned int mnTrackReferenceForFrame;
            
            // TrackLocalMap - searchByProjection 决定是否对isInFrustum判断的变量。
            // mnLastFrameSeen==mCurrentFrame.mnId 的情况。
            // a 已经和当前帧经过匹配(TrackReference, TrackWithMotionModel) 但在优化过程中认为时外点。
            // b 已经和当前帧经过进过匹配，内点，不需要投影。
            long unsigned int mnLastFrameSeen;

            // 用于local mapping的变量。
            long unsigned int mnBALocalForKF;
            long unsigned int mnFuseCandidateForKF;


            //  用于loop closing的变量。
            long unsigned int mnLoopPointForKF;
            long unsigned int mnCorrectedByKF;
            long unsigned int mnCorrectedReference;
            cv::Mat mPosGBA;
            long unsigned int mnBAGlobalForKF;


            static std::mutex mGlobalMutex;

        protected:

            cv::Mat mWorldPos;                              // MapPoint在世界坐标系下的绝对坐标。

            std::map<KeyFrame *, size_t> mObservations;       // 观测到该MapPoint的KF和该MapPoint在KF中的索引。

            cv::Mat mNormalVector;                          // MapPoint的平均观测方向。
            
            // 快速匹配最好的描述子。
            // 每个3D也有一个描述子。
            // 如果MapPoint与多帧图像特征点对应（由KF来构造时），那么距离其他描述子的平均距离最小的描述子是最佳描述子。
            // MapPoint只与一帧图像特征点对应（由Frame构造时），这个特征点的描述子就是该3D点的描述子。
            cv::Mat mDescriptor;                            // 通过 ComputeDistinctiveDescriptors()获得最佳描述子。

            KeyFrame *mpRefKF;                              // 参考关键帧。

            // 跟踪计数。
            int mnVisible;
            int mnFound;

            // 坏点标志（不会从内存中剔除MapPoint）。
            bool mbBad;
            MapPoint *mpReplaced;

            // 尺度不变性距离。
            float mfMinDistance;
            float mfMaxDistance;

            Map *mpMap;

            std::mutex mMutexPos;
            std::mutex mMutexFeatures;


    };


}       // namespace ORB_SLAM2

#endif 

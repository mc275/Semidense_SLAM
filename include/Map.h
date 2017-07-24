//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>

namespace ORB_SLAM2
{

    class MapPoint;
    class KeyFrame;

    class Map
    {
        public:
            // 默认构造函数。
            Map();
            
            // 在地图中插入关键帧。
            void AddKeyFrame(KeyFrame *pKF);
            // 在地图中插入点云。
            void AddMapPoint(MapPoint *pMP);
            // 在地图中剔除点云。
            void EraseMapPoint(MapPoint *pMP);
            // 在地图中剔除关键帧。
            void EraseKeyFrame(KeyFrame *pKF);
            // 设置参考MP，用于DrawMapPoints。
            void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

            // 获得所有关键帧。
            std::vector<KeyFrame *> GetAllKeyFrames();
            // 获得所有点云。
            std::vector<MapPoint *> GetAllMapPoints();
            // 获得参考点晕。
            std::vector<MapPoint *> GetReferenceMapPoints();


            long unsigned int MapPointsInMap();
            long unsigned  KeyFramesInMap();

            // 最后一帧关键帧？
            long unsigned int GetMaxKFid();

            void clear();

            std::vector<KeyFrame *> mvpKeyFrameOrigins;

            std::mutex mMutexMapUpdate;

            // 避免在不同线程中同时创建点云，造成Id冲突。
            std::mutex mMutexPointCreation;

        protected:

            std::set<MapPoint *> mspMapPoints;                  // 地图点云集。 
            std::set<KeyFrame *> mspKeyFrames;                  // 关键帧集。


            std::vector<MapPoint *> mvpReferenceMapPoints;      // 参考点云集。

            long unsigned int mnMaxKFid;                        // 最大帧Id。
            
            std::mutex mMutexMap;


    };

}   //namespace ORB_SLAM2

#endif

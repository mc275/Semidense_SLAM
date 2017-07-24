


#include "Map.h"
#include <mutex>

namespace ORB_SLAM2
{

    // 构造函数。
    Map::Map(): mnMaxKFid(0)
    {

    }



    // 在地图中插入关键帧 pKF。
    void Map::AddKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if(pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    // 在地图中插入地图点云 pMP。
    void Map::AddMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    // 从地图中剔除地图点云 pMP。
    void Map::EraseMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexMap);
        // 剔除指针。
        mspMapPoints.erase(pMP);
    }

    // 从地图中剔除关键帧 pKF。
    void Map::EraseKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    // 设置参考MapPoint，用于DrawMapPoint()画图。
    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
    {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    // 获取地图的所有关键帧。
    vector<KeyFrame *> Map::GetAllKeyFrames()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    // 获取地图的所有MapPoints。
    vector<MapPoint *> Map::GetAllMapPoints()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    // 获取地图中的点云数量。
    long unsigned int Map::MapPointsInMap()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    // 获取地图中关键帧数量。
    long unsigned int Map::KeyFramesInMap()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    // 获取地图中所有参考MapPoints。
    vector<MapPoint *> Map::GetReferenceMapPoints()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    // 获取地图中关键帧的最大Id。
    long unsigned int Map::GetMaxKFid()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    // 清除地图。
    void Map::clear()
    {
        // 释放MapPoints的内存。
        for(set<MapPoint *>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
            delete *sit;

        // 释放KeyFrame的内存。
        for(set<KeyFrame *>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
            delete *sit;

        mspMapPoints.clear();
        mspKeyFrames.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpKeyFrameOrigins.clear();
    }




}   // namespace ORB_SLAM2




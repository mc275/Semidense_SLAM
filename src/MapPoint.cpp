



#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>


namespace ORB_SLAM2
{

    // 静态成员变量定义，Id从0开始。
    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;


    /* 
    * 给定坐标与KeyFrame构造MapPoint。
    *   双目： StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()。
    *   单目： Tracking::CreateInitialMapMonocular(), LocalMapping::CreateNewMapPoints()。
    */
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap):
        mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
        mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3,1,CV_32F);

        // MapPoint在Tracking或Local Mapping线程创建，此处mutex防止id冲突。
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }



    /*
    * 给定坐标与Frame创建MapPing。
    *   双目： UpdateLastFrame()*
    */
    MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
        mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
        mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
        mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        cv::Mat Ow = pFrame->GetCameraCenter();
        // 世界坐标系下，相机指向3D点的向量。
        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector/cv::norm(mNormalVector);  // 归一化，单位向量。

        cv::Mat PC = Pos - Ow;
        const float dist = cv::norm(PC);
        const int level = pFrame->mvKeysUn[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels; 

        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

        // 左目特征点对应的描述子。
        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

        // 防止Id冲突。
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;

    } 



    // 设置MapPoint世界坐标系下的坐标。
    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);

    }

    // 获取地图点云世界坐标系下的坐标。
    cv::Mat MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    // 获取地图点的平均观测方向。
    cv::Mat MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    // 获取该地图点的参考关键帧。
    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    /*
    * 添加观测地图点云的关键帧，增加观测的相机数目nObs，单目+1，双目和RGBD+2。
    * 该函数时建立关键帧共视关系的核心函数。
    * @param
    *   pKF 关键帧。
    *   idx MapPoint在KeyFrame中的索引。
    */
    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 已经建立过观测关系。
        if(mObservations.count(pKF))
            return;

        // 记录下能观测到该Mappoint的KF和MapPoint在KF中的索引。
        mObservations[pKF] = idx;

        // 双目或RGBD
        if(pKF->mvuRight[idx] >= 0)
            nObs += 2;
        // 单目。
        else
            nObs++;

    }

    // 擦除关键帧pKF对该地图点云的观测关系。
    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                if(pKF->mvuRight[idx] >= 0)
                    nObs -= 2;
                else
                    nObs--;

                mObservations.erase(pKF);

                // 如果该关键帧是参考帧(创建地图点的关键帧)。
                if(mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                // 如果观测到该点云的相机数少于2，丢弃该点。
                if(nObs<=2)
                    bBad = true;
            }
        }

        if(bBad)
            SetBadFlag();

    }


    // 获取观测该点云的关键帧KF和该点云在KF中的索引。
    map<KeyFrame *, size_t> MapPoint::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    // 获取观测该点云的相机数目nObs。
    int MapPoint::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }



    // 设置该点为坏点，擦除可以观测到该MapPoint的所有关键帧与该MapPoint的关联关系。
    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            // 释放该MapPoint mObservations的内存空间。
            mObservations.clear();
        }

        for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            // 擦除pKF中索引号为mit->second的MapPoint。
            pKF->EraseMapPointMatch(mit->second);
        }

        // 释放该MapPoint在Map中的内存。
        mpMap->EraseMapPoint(this);

    }


    // 获得替换该地图点云的MapPoint。 
    MapPoint *MapPoint::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    // 形成闭环后，会更新KeyFrame与MapPoint的关系。
    void MapPoint::Replace(MapPoint *pMP)
    {
        if(pMP->mnId == this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            // 临时保存该地图点云与关键帧关联信息。
            obs = mObservations;
            mObservations.clear();
			mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        // 更新所有能观测到该MapPoint的KeyFrame的关联。
        for(map<KeyFrame *, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            // pKF之前没有观测到替换点云。
            if(!pMP->IsInKeyFrame(pKF))
            {
                // pKF替换地图点， pMP替换观测。
                pKF->ReplaceMapPointMatch(mit->second, pMP);
                pMP->AddObservation(pKF, mit->second);
            }
            // pKF之前也能观测到需要替换的点云，直接擦除当前点云即可。
            else
            {
                pKF->EraseMapPointMatch(mit->second);
            }
        }

        // 更新该点云的可以观察和找到的数量。
        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);
        pMP->ComputeDistinctiveDescriptors();

        // 在地图中剔除该MapPoint。
        mpMap->EraseMapPoint(this);

    }



    // 该MapPoint是否为坏点，true为坏点。
    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }



    /*
    * 增加可以观测到该地图点的帧数量。
    * 可以观测表示:
    *   1. 该MapPoint在某帧的视野范围内，通过Frame::IsInFrustum()判断。
    *   2. 该MapPoint可以被这些帧观测，但是不一定和这些帧的特征点匹配上。
    */
    void MapPoint::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    // 可以找到该点的帧数+n。
    void MapPoint::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    // 获得该地图点 被帧找到数/被帧观测数。
    float MapPoint::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures); 
        return static_cast<float>(mnFound)/mnVisible;
    }



    /*
    * 计算具有代表性的描述子。
    *   由于一个MapPoint可以被许多Frame观测到，在插入关键帧后，需要判断是否更新该地图点的最佳描述子。
    *   先获得所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小距离中值。
    */
    void MapPoint::ComputeDistinctiveDescriptors()
    {

        vector<cv::Mat> vDescriptors;
        map<KeyFrame *, size_t> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if(mbBad)
                return;
            observations = mObservations;
        }

        if(observations.empty())
            return;
        //分配内存。
        vDescriptors.reserve(observations.size());

        // 遍历观测到该MapPoint的所有关键帧，获得ORB描述子，插入到vDescriptors中。
        for(map<KeyFrame *, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            if(!pKF->isBad())
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if(vDescriptors.empty())
            return;

        // 获得描述子的数量。
        const size_t N = vDescriptors.size();
        // 描述子两两之间的距离。
        std::vector<std::vector<float> > Distances;
        Distances.resize(N, vector<float>(N,0));
        for(size_t i=0; i<N; i++)
        {
            // Distances[i][i]是描述子自身的距离。
            Distances[i][i]=0;
            for(size_t j=i+1; j<N; j++)
            {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // 
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0; i<N; i++)
        {
            // 第i个描述子到其他所有描述子的距离。
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            // 获得中值。
            int median = vDists[0.5*(N-1)];

            // 获得描述子去其他描述自的最小中值。
            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);

            // 最好描述子，该描述子相对于其他描述子有最小的中值距离。
            // 中值代表整个描述子到其他描述子的平距离。
            // 最好的描述子和其他描述子的平均距离最小。
            mDescriptor = vDescriptors[BestIdx].clone();
        }

    }



    // 获取MapPoint的描述子。
    cv::Mat MapPoint::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }



    // 获取该MapPoint在关键帧pKF的索引。
    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;

    }



    // 检查MapPoint是否被pKF观测到。
    bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }



    /*
    * 更新该MapPoint的平均观测方向和观测距离范围。
    *
    *   由于1个MapPoint会被许多相机观测到，因此插入关键帧后，需要更新相应的变量。
    *
    */
    void MapPoint::UpdateNormalAndDepth()
    {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;       
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if(mbBad)
                return;

            // 获取观测到该MapPoint的所有关键帧。
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos.clone();
        }

        if(observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
        int n=0;
        for(map<KeyFrame *, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            // 所有观测到该MapPoint的关键帧的观测向量归一化求和。
            normal = normal + normali/cv::norm(normali);
            n++;
        }

        // 在世界坐标系下，由参考帧相机指向地图点的向量。
        cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        const float dist = cv::norm(PC);
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;  // 金字塔层数。

        {
            unique_lock<mutex> lock3(mMutexPos);

            // 观测到该MapPoint的距离上限。
            mfMaxDistance = dist*levelScaleFactor;
            // 观测到该MapPoint的距离下限。
            mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
            // 平均观测方向，个人感觉不需要除n。
            mNormalVector = normal/n;
        }

    }



    // 获得该MapPoint最小不变性距离。
    float MapPoint::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f*mfMinDistance;
    }

    // 获得该MapPoint最大不变性距离。
    float MapPoint::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f*mfMaxDistance;
    }



    //              ____
    // Nearer      /____\     level:n-1 --> dmin
    //            /______\                       d/dmin = 1.2^(n-1-m)
    //           /________\   level:m   --> d
    //          /__________\                     dmax/d = 1.2^m
    // Farther /____________\ level:0   --> dmax
    //
    //           log(dmax/d)
    // m = ceil(------------)
    //            log(1.2) 
    int MapPoint::PredictScale(const float &currentDist, const float &logScaleFactor)
    {
        float ratio;
        {
            unique_lock<mutex> lock3(mMutexPos);
            // mfMaxDistance为参考帧考虑尺度后的距离。
            ratio = mfMaxDistance/currentDist;
        }

        // 取log线性化。
        return ceil(log(ratio)/logScaleFactor);

    }

}   // namespace ORB_SLAM2



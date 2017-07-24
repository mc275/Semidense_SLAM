//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "ProbabilityMapping.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{
    
    class Map;
    class MapPoint;
    class Frame;
    class KeyFrameDatabase;
    class ProbabilityMapping;
    
    

    // 关键帧，可以由Frame构造，许多数据会被3个线程同时访问，用锁的地方很普遍。

    class KeyFrame
    {
        public:

            // 构造函数。
            KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

            // 位姿函数。
            // 在这里的get set需要用到锁。
	    void SetPose(const cv::Mat &Tcw);
            cv::Mat GetPose();
            cv::Mat GetPoseInverse();
            cv::Mat GetCameraCenter();
            cv::Mat GetStereoCenter();
            cv::Mat GetRotation();
            cv::Mat GetTranslation();


            // 词袋表示。
            void ComputeBoW();

            //  Covisibility图函数。
            void AddConnection(KeyFrame *pKF, const int &weight);
            void EraseConnection(KeyFrame *pKF);
            void UpdateConnections();
            void UpdateBestCovisibles();
            std::set<KeyFrame *> GetConnectedKeyFrames();
            std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
            std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);
            std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);
            int GetWeight(KeyFrame *pKF);

            // Spanning树函数。
            void AddChild(KeyFrame *pKF);
            void EraseChild(KeyFrame *pKF);
            void ChangeParent(KeyFrame *pKF);
            std::set<KeyFrame *> GetChilds();
            KeyFrame *GetParent();
            bool hasChild(KeyFrame *pKF);

            // 闭环边。
            void AddLoopEdge(KeyFrame *pKF);
            std::set<KeyFrame *> GetLoopEdges();

            // 地图点云观察函数。
            void AddMapPoint(MapPoint *pMP, const size_t &idx);
            void EraseMapPointMatch(const size_t &idx);
            void EraseMapPointMatch(MapPoint *pMP);
            void ReplaceMapPointMatch(const size_t &idx, MapPoint * pMP);
            std::set<MapPoint *> GetMapPoints();
            std::vector<MapPoint *> GetMapPointMatches();
            int TrackedMapPoints(const int &minObs);
            MapPoint *GetMapPoint(const size_t &idx);

            // 特征点函数
            std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
            cv::Mat UnprojectStereo(int i);


            // 图像。
            bool IsInImage(const float &x, const float &y) const;

            // 使能/失效坏点标志。
            void SetNotErase();
            void SetErase();

            // 置位/检查坏点标志。
            void SetBadFlag();
            bool isBad();

            // 计算场景深度，在单目中使用。(q=2 中位数)
            float ComputeSceneMedianDepth(const int q);

            static bool weightComp(int a, int b)
            {
                return a>b;
            }

            static bool lId(KeyFrame *pKF1, KeyFrame *pKF2)
            {
                return pKF1->mnId < pKF2->mnId;
            }

            // 半稠密函数。
            cv::Mat GetImage();
	    cv::Mat GetCalibrationMatrix() const;
	    std::vector<float> GetAllPointDepths(int q = 2); 

	
	    // 以下变量只在一个线程中访问，或者从不更改。
	public:
	    
	    // nNextId改为nLastId比较好，表示上一个KF。
            static long unsigned int nNextId;
            
            // mnId为当前KF的Id,是nNextId+1。
            long unsigned int mnId;

            // KF的基本属性是一个Frame，初始化时需要Frame,
            // mnFrameId记录了该KF是哪个Frame初始化的。
            const long unsigned int mnFrameId;

            // 时间戳。
            const double mTimeStamp;

            // 栅格(加快特征匹配)。
            // 和Frame类中的定义相同。
            const int mnGridCols;
            const int mnGridRows;
            const float mfGridElementWidthInv;
            const float mfGridElementHeightInv;


            // 用于跟踪的变量。
            long unsigned int mnTrackReferenceForFrame; // 利用mCurrentKF.mnId作为标志位，避免3种不同加添局部关键的方法添加重复的关键帧。
            long unsigned int mnFuseTargetForKF;

            // 用于局部地图的变量。
            long unsigned int mnBALocalForKF;
            long unsigned int mnBAFixedForKF;

            // 用于关键帧数据库的变量。
            long unsigned int mnLoopQuery;
            int mnLoopWords;
            float mLoopScore;
            long unsigned int mnRelocQuery;
            int mnRelocWords;
            float mRelocScore;

            // 用于闭环检测的变量
            cv::Mat mTcwGBA;
            cv::Mat mTcwBefGBA;
            long unsigned int mnBAGlobalForKF;

            // 相机标定参数
            const float fx,fy,cx,cy,invfx,invfy,mbf,mb,mThDepth;

            // 特征点数量
            const int N;

            // 特征点，立体坐标系和描述子。通过一个索引关联。
            // 和Frame关联。
            const std::vector<cv::KeyPoint> mvKeys;
            const std::vector<cv::KeyPoint> mvKeysUn;
            const std::vector<float> mvuRight;      // 如果是单目，为负数。
            const std::vector<float> mvDepth;       // 如果是单目，为负数。
            const cv::Mat mDescriptors;

            // BoW。
            DBoW2::BowVector mBowVec;       // 图像的词袋表示。
            DBoW2::FeatureVector mFeatVec;  // 局部特征向量节点的索引。

            // 相对于父类的姿态（当坏点标志被激活后）。
            cv::Mat mTcp;

            // 尺度。
            const int mnScaleLevels;
            const float mfScaleFactor;
            const float mfLogScaleFactor;
            const std::vector<float> mvScaleFactors;    // 尺度因子， scale^n  scale=1.2 n为层数。
            const std::vector<float> mvLevelSigma2;     // 尺度因子的平方。
            const std::vector<float> mvInvLevelSigma2; 

            // 图像边界和标定
            const int mnMinX;
            const int mnMinY;
            const int mnMaxX;
            const int mnMaxY;
            const cv::Mat mK;       // 相机内参

            // 以下变量需要通过互斥体访问来保证线程安全。

        protected:

            // SE3位姿和相机质心。
            cv::Mat Tcw;
            cv::Mat Twc;
            cv::Mat Ow;

            cv::Mat Cw;             // 双目基线的中点，仅用于可视化。
            
            // 关联特征点的地图点云。
            std::vector<MapPoint *> mvpMapPoints;

            // BoW
            KeyFrameDatabase *mpKeyFrameDB;
            ORBVocabulary * mpORBvocabulary;

            // 覆盖在图像上的栅格。
            std::vector<std::vector <std::vector <size_t> > > mGrid;

            // Covisibility图。
            std::map<KeyFrame *, int> mConnectedKeyFrameWeights;        // 与该关键帧连接的关键和权重。
            std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;        // 排序后的关键帧。
            std::vector<int> mvOrderedWeights;                          // 排序后的权重，从大到小。

            // Spanning树和闭环边。
            // std::set是集合，和vector相比，插入数据时会自动排序。
            bool mbFirstConnection;
            KeyFrame *mpParent;
            std::set<KeyFrame *> mspChildrens;
            std::set<KeyFrame *> mspLoopEdges;

            // 坏点标志。
            bool mbNotErase;
            bool mbToBeErased;
            bool mbBad;

            float mHalfBaseline;    // 仅用于可视化。

            Map* mpMap;

            std::mutex mMutexPose;
            std::mutex mMutexConnections;
            std::mutex mMutexFeatures;
	    
	public:
	    // 半稠密参数。
	    cv::Mat im_;
	    cv::Mat rgb_;
	    bool semidense_flag_;  					// 当前关键帧是否用于半稠密地图构建。
	    cv::Mat GradImg,GradTheta;
	    double I_stddev;
	    cv::Mat depth_map_;
	    cv::Mat depth_sigma_;
	    cv::Mat SemiDensePointSets_;


    };


}   // namespace ORB_SLAM2

#endif

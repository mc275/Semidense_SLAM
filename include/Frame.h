//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef FRAME_H
#define FRAME_H

#include<vector>


#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"


#include <opencv2/opencv.hpp>


namespace ORB_SLAM2
{

    #define FRAME_GRID_ROWS 48
    #define FRAME_GRID_COLS 64
    

    class MapPoint;
    class KeyFrame;

    class Frame
    {
        
        public:
            
            // 默认构造函数，无任何实参的，执行默认初始化，函数体外类型=0，函数体内类型不初始化。
            Frame();

            // 拷贝构造函数，第一个参数必须是自身类类型，实际是把参数对象的值赋给定义对象，使用=表达式完成。
            Frame(const Frame &frame);



            // 双目构造函数
            Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

            // RGB-D构造函数
            Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

            // 单目构造函数
            Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);
			Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, cv::Mat rgbimg);
			
            // 提取图像中的ORB特征点，0表示左图像，1表示右图像。
            // 提取的点放在mvKeys中，描述子放在Descriptor中。
            // 直接用ORBextractor中的函数调用符提取特征呢。
            void ExtractORB(int flag, const cv::Mat &im);

            // 计算词袋表示，放在mBowVec中。
            void ComputeBoW();

            // 获取相机位姿，用Tcw更新mTcw。
            void SetPose(cv::Mat Tcw);


            // 根据相机的位姿计算旋转，平移和相机的中心矩。
            void UpdatePoseMatrices();

            // 返回相机中心。
            inline cv::Mat GetCameraCenter()
            {
                return mOw.clone();
            }

            // 获得旋转矩阵的逆矩阵。
            inline cv::Mat GetRotationInverse()
            {
                return mRwc.clone();
            }

            // 检查地图点云是否在相机的视野范围内,所有的点云变量是否满足跟踪要求。
            bool isInFrustum(MapPoint *pMP, float viewingCosLimit);


            // 计算特征点的单元格，如果在网格外返回false。
            bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);


            // 找到以(x,y)为中心，边长为2r的方形区域内，满足[minLevel,maxLevel]的特征点。
            vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel=-1, const int maxLevel=-1) const;

	    // 为左图的每个特征点在右图中寻找匹配的特征点。
            // 如果可以找到匹配，可以获得深度信息并且存储与左图对应的右图点的坐标。
            void ComputeStereoMatches();


            // 将图像中特征点与深度地图的坐标相关联。
            void ComputeStereoFromRGBD(const cv::Mat &imDepth);


            // 将特征点(如果立体匹配或深度信息有效)反向投影到3D世界坐标系下。
            cv::Mat UnprojectStereo(const int &i);


        public:

            // 用于重定位的词典。
            ORBVocabulary *mpORBvocabulary;


            // 提取特征，双目是右图才有用。
            ORBextractor *mpORBextractorLeft, *mpORBextractorRight;


            // 帧图像的时间标签。
            double mTimeStamp;

            // 相机标定参数和OpenCV的畸变参数。
            cv::Mat mK;
            static float fx;
            static float fy;
            static float cx;
            static float cy;
            static float invfx;
            static float invfy;
            cv::Mat mDistCoef;
	    

            // 立体摄像头基线与fx的乘积。
            float mbf;


            // 双目基线(m)
            float mb;

            // 远近点深度的阈值。近点从视角1插入。在单目情况下原点从视角2插入。
            float mThDepth;


            // 特征点的数量。
            int N;


            // 特征点描述系向量(视图中的原始情况)和修正后的特征点描述向量(系统实际使用)。
            // 在双目下，mvKeysUn是冗余的，因为一般得到的图像已经经过校正。
            // 在RGBD下，RGB图像可能失真。
            // mvKeys, mvKeysRight是左右两幅图像提取的特征点(未校正)。
            // mvKeysUn 时经过校正后的特征点。
            std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
            std::vector<cv::KeyPoint> mvKeysUn;


            // 对于立体相机，mvuRight是特征点的右图坐标，mvDepth是特征点的深度。
            // 对于单目相机，这两个容器的值是-1。
            std::vector<float> mvuRight;
            std::vector<float> mvDepth;


            // 词袋向量的结构
            DBoW2::BowVector mBowVec;
            DBoW2::FeatureVector mFeatVec;


            // ORB特征点的描述子，每一行表示表示一个特征点的描述子。
            cv::Mat mDescriptors, mDescriptorsRight;

            // 每个特征点对应的地图点云，NULL表示没有对应的点云。
            std::vector<MapPoint *> mvpMapPoints;


            // 在地图中观测不到的3D点。
            std::vector<bool> mvbOutlier;


            // 投影MapPoints时，把特征点分配给不同的网格的单元格减少匹配的复杂性。
            // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv可以确定哪个格子。
            static float mfGridElementWidthInv;
            static float mfGridElementHeightInv;


            // 每个各自分配的特征点数，把图像分成格子，保证均匀提取特征。
            // FRAME_GRID_ROWS 48
            // FRAME_GRID_COLS 64
            std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];


            // 相机位姿 更新时的位姿变换矩阵是从世界坐标系到相机坐标系的变换矩阵。
            cv::Mat mTcw;

            // 当前帧和下一帧的Id。
            static long unsigned int nNextId;
            long unsigned int mnId;

            // 参考关键帧,一般指的就是当前的关键帧。
            KeyFrame *mpReferenceKF;

            // 尺度金字塔信息。
            int mnScaleLevels;      // 图像金字塔层数
            float mfScaleFactor;    // 图像金字塔尺度信息
            float mfLogScaleFactor;
            vector<float> mvScaleFactors;
            vector<float> mvInvScaleFactors;
            vector<float> mvLevelSigma2;
            vector<float> mvInvLevelSigma2;
	    
	    cv::Mat im_;
	    cv::Mat rgb_;

            // 无畸变图像边界 栅格边界
            static float mnMinX;
            static float mnMaxX;
            static float mnMinY;
            static float mnMaxY;


            // 初始化？
            static bool mbInitialComputations;



        private:

            // 通过OpenCV的畸变参数，给出无畸变特征点。
            // 仅对于RGB-D,立体摄像头已经进行了校正。
            // 在构造函数中调用。
            void UndistortKeyPoints();


            // 对无畸变的图像计算图像边界。
            // 在构造函数中调用。
            void ComputeImageBounds(const cv::Mat &imLeft);

            // 为网格分配特征点，加快特征匹配。
            // 在构造函数中调用。
            void AssignFeaturesToGrid();

            // 旋转，平移和相机质心。
            cv::Mat mRcw;   // 从世界到相机坐标系的旋转。
            cv::Mat mtcw;   // 从世界到相机坐标系的平移。
            cv::Mat mRwc;   // 从相机到世界坐标系的旋转。
            cv::Mat mOw;    // mtwc,从相机到世界坐标系的平移。



    };


}

#endif


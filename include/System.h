//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include"Tracking.h"
#include"FrameDrawer.h"
#include"MapDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"KeyFrameDatabase.h"
#include"ORBVocabulary.h"
#include"Viewer.h"
#include "ProbabilityMapping.h"

namespace ORB_SLAM2
{
   
class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;
    
class System
{
    public:
        // 输入传感器。
        enum eSensor
        {
            MONOCULAR=0,
            STEREO=1,
            RGBD=2
        };
    

    public:
        // 构造函数，初始化SLAM系统，运行局部地图，回路闭合和地图显示线程。
        System( const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true );

        // 处理双目采集图像。
        // 输入：RGB或者灰度,两幅图像需要保持同步；时间戳。        输出：返回相机位姿。
        cv::Mat TrackStereo( const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp );
    
        // 处理RGBD图像
        // 输入：RGB图像和深度信息，需要保持对应关系；时间戳。     输出：返回相机位姿。
        cv::Mat TrackRGBD( const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp );
    
        // 处理单目采集图像
        // 输入：RGB图像或灰度图像；时间戳。                       输出：返回相机位姿。
        cv::Mat TrackMonocular( const cv::Mat &im, const double &timestamp );


        // 停止局部地图线程，只进行相机跟踪。
        void ActivateLocalizationMode();

        // 恢复局部地图线程。
        void DeactivateLocalizationMode();

        // 重置SLAM系统，清除地图。
        void Reset();

        // 要求终止所有线程。
        // 当所有线程运行完成后才会终止线程。
        // 需要在轨迹存储前进行调用。
        void Shutdown();

        // 按照Tum数据集格式的保存相机轨迹。
        // 首先需要调用终止线程。
        // 具体数据格式见 http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveTrajectoryTUM( const string &filename) ;

        // 按照TUM数据集格式保存关键帧相机的位姿。
        // 这个函数用于单目传感器。
        // 调用前需要调用ShutDown函数
        // 数据格式见 http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveKeyFrameTrajectoryTUM( const string &filename );

        // 按照KITTI数据集格式保存相机轨迹。
        // 首先需要调用ShutDown函数
        // 数据格式见 http://www.cvlibs.net/datasets/kitti/eval_odometry.php
	void SaveTrajectoryKITTI(const string &filename);

        // 未来的工作
        //SaveMap( const string &filename );
        //LoadMap( const string &filename )


    private:

        // 输入传感器。
        eSensor mSensor;

        // 用于位置识别的特征匹配ORB词典库。
        ORBVocabulary *mpVocabulary;

        // 用于位置识别的关键帧数据库（重定位和闭环监测）。
        KeyFrameDatabase *mpKeyFrameDatabase;

        // 地图结构，存储所有关键帧和地图点云
        Map *mpMap;

        // 跟踪类。接收一帧图像并且计算相机位姿。
        // 也可以决定是否插入关键帧，创建新的地图点云。
        // 当跟踪丢失后，可以进行重定位
        Tracking *mpTracker;

        // 局部地图类。管理局部地图，运行局部BA。
        LocalMapping *mpLocalMapper;

        // 闭环检测类。从每一个新的关键帧中搜索闭环。 
        // 如果找到一个闭环，运行pose图优化并在新的线程中运行全BA。
        LoopClosing *mpLoopCloser;

        // 建立视图模式，画出地图和当前的相机位姿。使用Pangolin。
        Viewer *mpViewer;

        FrameDrawer *mpFrameDrawer;
        MapDrawer *mpMapDrawer;
		ProbabilityMapping* mpSemiDenseMapping;
		
        // 系统线程：局部地图，闭环监测，视图显示。
        // 跟踪线程在创建系统对象的主线程中。
        std::thread *mptLocalMapping;
        std::thread *mptLoopClosing;
        std::thread *mptViewer;
		std::thread* mptSemiDense;
		
        // 重置标志
        std::mutex mMutexReset;
        bool mbReset;

        //模式切换标志
        std::mutex mMutexMode;
        bool mbActivateLocalizationMode;
        bool mbDeactivateLocalizationMode;



};


} //namespace ORB_SLAM2;


#endif //SYSTEM_H



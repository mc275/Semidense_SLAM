//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{
    
    // ORB中以cv::Mat为基本存储结构，但是在g2o和Eigen中需要转换。


    class Converter
    {
        public:

            // 一个描述子矩阵到一串单行描述子向量。
            static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

            // cv::Mat/g2o::Sim3 to g2o::SE3Quat.
            static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
            // 未实现。
            static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);


            // g2o Eigen中的存储结构 to cv::Mat
            static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
            static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
            static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
            static cv::Mat toCvMat(const Eigen::Matrix3d &m);
            static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
            static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);


            // cv to Eigen
            static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
            static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
            static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
            static std::vector<float> toQuaternion(const cv::Mat &M);

    
    };

}


#endif


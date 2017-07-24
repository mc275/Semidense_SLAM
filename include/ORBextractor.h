//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{
    
    class ExtractorNode
    {
        public:

            // 定义构造函数，列表初始化成员变量。
            ExtractorNode():bNoMore(false)
            { }

            void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

            std::vector<cv::KeyPoint> vKeys;
            cv::Point2i UL,UR,BL,BR;
            std::list<ExtractorNode>::iterator lit;
            bool bNoMore;

    };

    class ORBextractor
    {
        public:

            enum 
            {
                HARRIS_SCORE=0,
                FAST_SCORE=1
            };

            ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

            ~ORBextractor(){}


            // 计算图像中ORB特征和描述子。
            // ORB被八叉树分散在图像中。
            // 在当前应用中忽略mask。
            // operator 是函数调用运算符，通过直接调用类定义的对象配合形参来调用这个函数，例如 ORBextractor *orbextractor;  (*orbextractor)(cv::InputArray image, cv::InputArray mask, 
            // std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors)
            void operator() ( cv::InputArray image, cv::InputArray mask, 
                    std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors );
            
            int inline GetLevels()
            {
                return nlevels;
            }

            float inline GetScaleFactor()
            {
                return scaleFactor;
            }

            std::vector<float> inline GetScaleFactors()
            {
                return mvScaleFactor;
            }

            std::vector<float> inline GetInverseScaleFactors()
            {
                return mvInvScaleFactor;
            }

            std::vector<float> inline GetScaleSigmaSquares()
            {
                return mvLevelSigma2;
            }

            std::vector<float> inline GetInverseScaleSigmaSquares()
            {
                return mvInvLevelSigma2;
            }
            

            std::vector<cv::Mat> mvImagePyramid;


        protected:

            void ComputePyramid(cv::Mat image);
            void ComputeKeyPointsOctTree(std::vector< std::vector<cv::KeyPoint> > &allKeypoints);
            std::vector<cv::KeyPoint> DistributeOctTree( const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &minY, const int &maxY,const int &nFeatures, const int &level );
            

            void ComputeKeyPointsOld( std::vector< std::vector<cv::KeyPoint> > &allKeypoints );

            std::vector<cv::Point> pattern;


            int nfeatures;
            double scaleFactor;
            int nlevels;
            int iniThFAST;
            int minThFAST;

            std::vector<int> mnFeaturesPerLevel;

            std::vector<int> umax;

            std::vector<float> mvScaleFactor;
            std::vector<float> mvInvScaleFactor;
            std::vector<float> mvLevelSigma2;
            std::vector<float> mvInvLevelSigma2;

            








            



    };






}

#endif


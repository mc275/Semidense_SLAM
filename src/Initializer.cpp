





#include "Initializer.h"
#include "Optimizer.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include <thread>

namespace ORB_SLAM2
{

    // @brief 构造函数，给定参考帧构造Initializer。
    // @param 
    //      ReferenceFrame    参考帧，SLAM正式开始的第一帧。
    //      sigma             测量误差。
    //      iterations        RANSAC迭代次数。
    Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
    {
        mK = ReferenceFrame.mK.clone();

        mvKeys1 = ReferenceFrame.mvKeysUn;

        mSigma = sigma;
        mSigma2 = sigma*sigma;
        mMaxIterations = iterations;
    }




    // 并行计算单应矩阵和基本矩阵，恢复出最开始两帧之间的相对位姿和三角化点云。
    bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                            vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
    {
        // ReferenceFrame: 1, CurrentFrame: 2。
        // Frame2的特征点。 
        mvKeys2 = CurrentFrame.mvKeysUn;

        // 1 2帧间匹配的特征点。
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());
        // 1帧的特征点在2帧中是否有匹配关键点。
        mvbMatched1.resize(mvKeys1.size());
	
	// 步骤1 特征点配对设置。
	for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
	{
	  // 第1 2帧之间有匹配。vMatches12是第2帧中与第1帧特征点匹配的特征点索引号。
	  if(vMatches12[i]>=0)
	  {
	      mvMatches12.push_back(make_pair(i, vMatches12[i]));
	      mvbMatched1[i] = true;
	  }
	  else
	    // 第1帧在第2帧中无匹配。
	    mvbMatched1[i] = false;
	}

    // 匹配的特征点个数。
    const int N = mvMatches12.size();

    // 容器vAllIndices，生成0到N-1的数作为索引。
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // 步骤2 在第1帧和第二帧的所有配对的特征点对中，随机选择8对匹配的特征点为一组，共mMaxIterations(200)组。用于F和H矩阵的求解。
    mvSets = vector< vector<size_t> >(mMaxIterations, vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        for(size_t j=0; j<8; j++)
        {
            // 产生０到N-1的随机数。
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);
            // idx表示随机产生的特征点索引号。
            int idx = vAvailableIndices[randi];

            // 行表示迭代次数，列表示每次迭代所需８对匹配点的序号。mvSets存储对应特征点的索引。
            mvSets[it][j] = idx;

            // randi对应的索引用最后一个元素替代。
            vAvailableIndices[randi] = vAvailableIndices.back();
            // 从容器中剔除最后一个元素，容器大小变小，循环产生的随机数不会重复抽取选择过的索引randi。
            vAvailableIndices.pop_back();
        }
    }

    // 步骤３双线程并行计算矩阵H和F。
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;           // 两个矩阵的得分。
    cv::Mat H, F;           // 单应矩阵和本征矩阵。

    // ref()　表示引用＆
    // 计算单应矩阵H和分数SH。
    thread threadH(&Initializer::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H) );

    // 计算基本矩阵F和分数SF。
    thread threadF(&Initializer::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F) );

    // 等待两个线程完成运算。
    threadH.join();
    threadF.join();

    // 步骤4 计算得分比例。
    float RH = SH/(SH+SF);
    
    // 步骤5 选择合适的初始化模型，恢复R,t。
    if(RH > 0.40)
        return ReconstructH(vbMatchesInliersH, H, mK, R21, t21, vP3D, vbTriangulated, 1.0, 50);
    else 
        return ReconstructF(vbMatchesInliersF, F, mK, R21, t21, vP3D, vbTriangulated, 1.0, 50);

    return false;

    }



    // 计算单应矩阵，假设场景为平面通过前两针求取H21，并得到该模型的评分。
    void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
    {

        // 两帧间的匹配特征点数量。
        const int N = mvMatches12.size();

        // 把mvKeys1和mvKeys2归一化同一尺度，均值为0, 一阶绝对矩为１，归一化后的矩阵为T1, T2。
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2inv = T2.inv();

        score = 0.0;                                    // 初始化模型得分。
        vbMatchesInliers = vector<bool> (N, false);     // 特征点匹配质量。

        // RANSAC迭代匹配特征点坐标。
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat H21i, H12i;
        // 每次RANSAC所有特征点的匹配情况和得分。
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        // RANSAC迭代，保存最高分。
        for(int it=0; it<mMaxIterations; it++)
        {
            // 提取随机选取的８对匹配特征点mvSets的坐标。
            for(size_t j=0; j<8; j++)
            {
                int idx = mvSets[it][j];    // 匹配特征点索引号。
                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Hn = ComputeH21(vPn1i, vPn2i);

            // 恢复原始的均值和尺度。
            H21i = T2inv*Hn*T1;
            H12i = H21i.inv();

            // 利用冲投影误差对RANSAC评分。
            currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

            // 获得用于RANSAC的匹配特征点的匹配质量和初始化模型的分数。
            if(currentScore > score)
            {
                H21 = H21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }

    }


    // 计算基础矩阵F。
    // 假设场景非平面，通过前两帧求取F。
    void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
    {
        // 两帧间匹配的特征点数量。
        const int N = vbMatchesInliers.size();
        
        // 归一化坐标, 统一尺度。
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2t = T2.t();

        // 特征点匹配质量和初始化模型分数。
        score = 0.0;
        vbMatchesInliers = vector<bool> (N,false);

        // ８对匹配特征点的坐标。
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat F21i;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        // 进行RANSAC迭代并保存最高分数。
        for(int it = 0; it < mMaxIterations; it++)
        {
            // 遍历迭代的一组８个匹配特征点。
            for(int j=0; j<8; j++)
            {
                int idx = mvSets[it][j];
                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

            F21i = T2t*Fn*T1;

            // 利用重投影误差对RANSAC结果评分。
            currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

            if(currentScore > score)
            {
                F21 = F21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }



    // |x'|     | h1 h2 h3 ||x|
    // |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子。
    // |1 |     | h7 h8 h9 ||1|
    // 使用DLT(direct linear tranform)求解该模型。
    // x' = a H x 。
    // ---> (x') 叉乘 (H x)  = 0
    // ---> Ah = 0
    // A = | 0  0  0 -x -y -1  xy'  yy'  y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
    //     | x  y  1  0  0  0 -xx' -yx' -x'|
    // 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解 。
    
    /**
    *       根据匹配特征点求解单应矩阵H(DLT)。具体理论见多视几何中的P68, 算法3.2。
    *　Param
    *       vP1 参考帧(Frame 1)归一化的点坐标。
    *       vP2 当前帧(Frame 2)归一化的点坐标。
    *　return
    *       单应矩阵。
    */
    cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(2*N, 9, CV_32F);  // 矩阵大小2Nx9。

        for(int i =0; i< N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            // 某对匹配特征点在矩阵A中的第一行。
            A.at<float>(2*i, 0) = 0.0;
            A.at<float>(2*i, 1) = 0.0;
            A.at<float>(2*i, 2) = 0.0;
            A.at<float>(2*i, 3) = -u1;
            A.at<float>(2*i, 4) = -v1;
            A.at<float>(2*i, 5) = -1;
            A.at<float>(2*i, 6) = v2*u1;
            A.at<float>(2*i, 7) = v2*v1;
            A.at<float>(2*i, 8) = v2;

            // 第二行。
            A.at<float>(2*i+1, 0) = u1;
            A.at<float>(2*i+1, 1) = v1;
            A.at<float>(2*i+1, 2) = 1;
            A.at<float>(2*i+1, 3) = 0.0;
            A.at<float>(2*i+1, 4) = 0.0;
            A.at<float>(2*i+1, 5) = 0.0;
            A.at<float>(2*i+1, 6) = -u2*u1;
            A.at<float>(2*i+1, 7) = -u2*v1;
            A.at<float>(2*i+1, 8) = -u2;
        }

        cv::Mat u, w, vt;

        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0,3);  // 解为vt的最后一列。

    }


    // x' = Fx =>　x'Fx = 0 => Af = 0
    // A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
    // 通过SVD分解，A'A最小特征值对应的特征向量为解。具体理论见多视几何P191。

    /*
    *   从特征点匹配求解基本矩阵F。
    * Param
    *   vP1 Frame1中的归一化特征点坐标。
    *   vP2 Frame2中的归一化特征点坐标。
    * return
    *   基本矩阵。　
    */
    cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();
        cv::Mat A(N, 9, CV_32F);    // 矩阵大小N×9。
        
        for(int i=0; i<N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            // 某个特征点坐标对应的矩阵A的一行。
            A.at<float>(i, 0) = u2*u1;
            A.at<float>(i, 1) = u2*v1;
            A.at<float>(i, 2) = u2;
            A.at<float>(i, 3) = v2*u1;
            A.at<float>(i, 4) = v2*v1;
            A.at<float>(i, 5) = v2;
            A.at<float>(i, 6) = u1;
            A.at<float>(i, 7) = v1;
            A.at<float>(i,8 ) = 1;
        }

        cv::Mat u, w, vt;

        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        cv::Mat Fpre = vt.row(8).reshape(0, 3);     // vt的最后一列。

        cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        w.at<float>(2) = 0; // 秩为２，第三个奇异值设置为０。

        return u*cv::Mat::diag(w)*vt;

    }



    // 对求出的单应矩阵H打分。
    // 具体理论　多视几何　P57 3.2.2几何距离；P73 3.7.1 RANSAC。
    float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
    {
        // 匹配特征点数量。
        const int N = mvMatches12.size();

        // |h11 h12 h13|
        // |h21 h22 h23|
        // |h31 h32 h33|
        const float h11 = H21.at<float>(0,0);
        const float h12 = H21.at<float>(0,1);
        const float h13 = H21.at<float>(0,2);
        const float h21 = H21.at<float>(1,0);
        const float h22 = H21.at<float>(1,1);
        const float h23 = H21.at<float>(1,2);
        const float h31 = H21.at<float>(2,0);
        const float h32 = H21.at<float>(2,1);
        const float h33 = H21.at<float>(2,2);

        // |h11inv h12inv h13inv|
        // |h21inv h22inv h23inv|
        // |h31inv h32inv h33inv|
        const float h11inv = H12.at<float>(0,0);
        const float h12inv = H12.at<float>(0,1);
        const float h13inv = H12.at<float>(0,2);
        const float h21inv = H12.at<float>(1,0);
        const float h22inv = H12.at<float>(1,1);
        const float h23inv = H12.at<float>(1,2);
        const float h31inv = H12.at<float>(2,0);
        const float h32inv = H12.at<float>(2,1);
        const float h33inv = H12.at<float>(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;

        // 基于卡方分布计算的阈值。
        const float th = 5.991;

        // 信息矩阵，方差平方的倒数。
        const float invSigmaSquare = 1.0/(sigma*sigma);

        // N对特征匹配点。
        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            //　把第二帧中的图像投影到第一帧。
            //  x2in1 = H12 * x2;
            // |u1|          |h11inv h12inv h13inv||u2|
            // |v1|        = |h21inv h22inv h23inv||v2|
            // |w2in1inv |   |h31inv h32inv h33inv||1 |
            const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
            const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
            const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

            // 计算冲投影误差。
            const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

            // 根据方差归一化误差。
            const float chiSquare1 = squareDist1*invSigmaSquare;

            if(chiSquare1 > th)
                bIn = false;
            else 
                score += th - chiSquare1;

            // 把第１帧中的特征点投影到第二帧中。
            // x1in2 = H21 * x1
            // 将图像１中的特征点单应到图像２中。
            const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
            const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
            const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

            // 重投影误差。
            const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

            // 方差归一化。
            const float chiSquare2 = squareDist2*invSigmaSquare;

            if(chiSquare2 > th)
                bIn = false;
            else 
                score += th - chiSquare2;

            if(bIn)
                vbMatchesInliers[i] = true;
            else
                vbMatchesInliers[i] = false;
        }
        return score;

    }



    // 对求出的基本矩阵F打分。
    float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        const float f11 = F21.at<float>(0,0);
        const float f12 = F21.at<float>(0,1);
        const float f13 = F21.at<float>(0,2);
        const float f21 = F21.at<float>(1,0);
        const float f22 = F21.at<float>(1,1);
        const float f23 = F21.at<float>(1,2);
        const float f31 = F21.at<float>(2,0);
        const float f32 = F21.at<float>(2,1);
        const float f33 = F21.at<float>(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;

        // 基于卡方分布的检验阈值。
        const float th = 3.841;
        const float thScore = 5.991;
        
        // 信息矩阵。
        const float invSigmaSquare = 1.0/(sigma*sigma);

        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // 计算第１帧图像到第２帧图像的冲投影误差。
            // l2 = F21 ×　x1 计算出第１帧图像中的特征点对应的过第２帧图像匹配点的对极线。
            const float a2 = f11*u1+f12*v1+f13;
            const float b2 = f21*u1+f22*v1+f23;
            const float c2 = f31*u1+f32*v1+f33;

            // 第２帧的匹配点在l2上，点乘=0。
            const float num2 = a2*u2+b2*v2+c2;

            const float squareDist1 = num2*num2/(a2*a2+b2*b2);    // 点到线的距离的平方

            const float chiSquare1 = squareDist1*invSigmaSquare;

            if(chiSquare1 > th)
                bIn = false;
            else
                score += thScore - chiSquare1;

            // 计算第２帧到第１帧的重投影误差。
            // l1 = F12 ×　x2
            const float a1 = f11*u2+f21*v2+f31;
            const float b1 = f12*u2+f22*v2+f32;
            const float c1 = f13*u2+f23*v2+f33;

            const float num1 = a1*u1+b1*v1+c1;

            const float squareDist2 = num1*num1/(a1*a1+b1*b1);

            const float chiSquare2 = squareDist2*invSigmaSquare;

            if(chiSquare2 > th)
                bIn = false;
            else
                score += thScore - chiSquare2;

            if(bIn)
                vbMatchesInliers[i] = true;
            else
                vbMatchesInliers[i] = false;
        }

        return score;

    }



    /** 
    *   根据基本矩阵F恢复R,t。
    *   １．通过相机内参，得到本征矩阵E=K'.t*F*K。
    *   ２．SVD分解，得到４组R,t。
    *   ３．带入进行深度检验，得到合适解。
    *   具体理论见多视几何，P175结论8.19.
    */
    bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21, 
                                    cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {

        int N=0;
        for(size_t i=0, iend=vbMatchesInliers.size(); i<iend; i++)
        {
            if(vbMatchesInliers[i])
                N++;
        }

        // 计算本征矩阵。
        cv::Mat E21 = K.t()*F21*K;

        cv::Mat R1, R2, t;

        // 对t进行了归一化，但是不决定整个SLAM的尺度。
        // 因为Tracking::CreateInitialMapMonocular函数对3D点深度进行了放缩，影响t的尺度。
        DecomposeE(E21, R1, R2, t);

        cv::Mat t1 = t;
        cv::Mat t2 = -t;

        // 进行带入检验，深度为正时的解为合适的解。
        vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        // 返回可以进行三角化的点个数。
        int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

        int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

        R21 = cv::Mat();
        t21 = cv::Mat();

        // minTriangulated为可以三角化恢复的三维点云个数。
        int nMinGood = max(static_cast<int>(0.9*N), minTriangulated);

        int nsimilar = 0;
        if(nGood1>0.7*maxGood)
            nsimilar++;
        if(nGood2>0.7*maxGood)
            nsimilar++;
        if(nGood3>0.7*maxGood)
            nsimilar++;
        if(nGood4>0.7*maxGood)
            nsimilar++;

        // 四个结果中，没有明显最优结果，返回失败。
        if(maxGood < nMinGood || nsimilar > 1)
            return false;

        // 比较大的视差。
        if(maxGood == nGood1)
        {
            if(parallax1 > minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                R1.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if(maxGood == nGood2)
        {
            if(parallax2 > minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                R2.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if(maxGood == nGood3)
        {
            if(parallax3 > minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                R1.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }
        else if(maxGood == nGood4)
        {
            if(parallax4 > minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                R2.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }

        return false;
    }



    // 从单应矩阵H中恢复R,t。
    // 单应矩阵常见的分解方法两种，Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition。
    // 本段代码采用Faugeras SVD-based decomposition。
    bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K, 
                                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        int N=0;
        for(size_t i=0, iend=vbMatchesInliers.size(); i<iend; i++) 
        {
            if(vbMatchesInliers[i])
                N++;
        }

        // H进行分解。
        cv::Mat invK = K.inv();
        cv::Mat A = invK*H21*K;

        cv::Mat U, w, Vt, V;
        cv::SVD::compute(A, w,U, Vt, cv::SVD::FULL_UV);
        V=Vt.t();

        float s = cv::determinant(U)*cv::determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        // SVD分解的特征值一般降序排列。
        if(d1/d2<1.00001 || d2/d3<1.00001)
        {
            return false;
        }

        vector<cv::Mat> vR, vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
        float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};
        float x3[] = {aux3, -aux3, aux3, -aux3};

        float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

        float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        // 计算R'
        for(int i=0; i<4; i++)
        {
            cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
            Rp.at<float>(0,0) = ctheta;
            Rp.at<float>(0,2) = -stheta[i];
            Rp.at<float>(2,0) = stheta[i];
            Rp.at<float>(2,2) = ctheta;

            cv::Mat R = s*U*Rp*Vt;
            vR.push_back(R);

            cv::Mat tp(3,1,CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = -x3[i];
            tp *= d1-d3;

            // 对t进行归一化，但Tracking::CreateInitialMapMonocular对3D点深度缩放，影响t的尺度。
            cv::Mat t = U*tp;
            vt.push_back(t/cv::norm(t));

            cv::Mat np(3,1,CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V*np;
            if(n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

        float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        // 计算旋转矩阵R'。
        for(int i=0; i<4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3,3,CV_32F);
            Rp.at<float>(0,0) = cphi;
            Rp.at<float>(0,2) = sphi[i];
            Rp.at<float>(1,1) = -1;
            Rp.at<float>(2,0) = sphi[i];
            Rp.at<float>(2,2) = -cphi;

            cv::Mat R = s*U*Rp*Vt;
            vR.push_back(R);

            cv::Mat tp(3,1,CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = x3[i];
            tp *= d1+d3;

            cv::Mat t = U*tp;
            vt.push_back(t/cv::norm(t));

            cv::Mat np(3,1,CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V*np;
            if(n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        vector<cv::Point3f> bestP3D;
        vector<bool> bestTriangulated;

        // ８组R,t。
        for(size_t i=0; i<8; i++)
        {
            float parallaxi;
            vector<cv::Point3f> vP3Di;
            vector<bool> vbTriangulatedi;
            int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

            // 保留最优和次优的。
            if(nGood > bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;
                bestP3D = vP3Di;
                bestTriangulated = vbTriangulatedi;
            } 
            else if(nGood > secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
        {
            vR[bestSolutionIdx].copyTo(R21);
            vt[bestSolutionIdx].copyTo(t21);
            vP3D = bestP3D;
            vbTriangulated = bestTriangulated;

            return true;
        }

        return false;

    }



    // Trianularization: 已知匹配特征点对{x x'} 和各自相机矩阵{P P'}, 估计三维点 X
    // x' = P'X  x = PX
    // 它们都属于 x = aPX
    //                         |X|
    // |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--|
    // |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--| X
    // |z|     |p9 p10 p11 p12||1|     |z|    |--p2--|
    // 采用DLT的方法：x×PX = 0
    // |yp2 -  p1|     |0|
    // |xp2  - p0| X = |0|
    // |xp1 - yp0|     |0|
    // 两个点
    // 变成程序中的形式：
    // |xp2  - p0 |     |0|
    // |yp2  - p1 | X = |0| ===> AX = 0
    // |x'p2'- p0'|     |0|
    // |y'p2'- p1'|     |0|
    // 具体理论见多视几何P217 11.2节。

    // 给定相机矩阵P,P'和图像上的特征点kp1,kp2，恢复世界坐标系下的3D坐标。   
    void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
    {
        // 之前的Decompose和Reconstruct函数对t有归一化，
        // 三角花恢复的3D点深度和t的尺度有关，
        // 但是由于Tracking::CreateInitialMapMonocular对3D点深度会缩放，这里的t不是SLAM的尺度值。

        cv::Mat A(4, 4, CV_32F);

        A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
        A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
        A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
        A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        // 归一化。
        x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
    
    }



    /**
    *   归一化特征点到同一尺度。
    *   [x',y',1]' = T*[x, y, 1]
    *   归一化后x',y'均值为０,　sum(abs(x_i'-0))=1, sum(abs(y_i'-0))=1
    *   
    * param
    *   vKeys　             特征点在图像上的坐标。
    *   vNormalizedPoints   特征点归一化后的坐标。
    *   T                   将特征点归一化的矩阵。
    */
    void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
    {
        float meanX = 0;
        float meanY = 0;
        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for(int i=0; i<N; i++)
        {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }

        meanX = meanX/N;
        meanY = meanY/N;

		float meanDevX = 0;
		float meanDevY = 0;
        // 将所有vKeys点就减去中心坐标，x坐标和y坐标的均值为0。
        for(int i=0; i<N; i++)
        {
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX/N;
        meanDevY = meanDevY/N;

        float sX = 1.0/meanDevX;
        float sY = 1.0/meanDevY;

        // 将x坐标和y坐标分别进行尺度放缩，使得x,y坐标的绝对矩为１。
        for(int i=0; i<N; i++)
        {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        // |sX  0  -meanx*sX|
        // |0   sY -meany*sY|
        // |0   0      1    |
        T = cv::Mat::eye(3,3,CV_32F);
        T.at<float>(0,0) = sX;
        T.at<float>(1,1) = sY;
        T.at<float>(0,2) = -meanX*sX;
        T.at<float>(1,2) = -meanY*sY;

    }



    // 对给出的基本矩阵F或单应矩阵H进行检验，找到合适的R,t。
    int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                                const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, 
                                const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
    {
        // 相机标定参数。
        const float fx = K.at<float>(0,0);
        const float fy = K.at<float>(1,1);
        const float cx = K.at<float>(0,2);
        const float cy = K.at<float>(1,2);

        vbGood = vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // 步骤１　得到一个相机的相机矩阵 P1=K[I|0]。
        // 以第一个相机的光心作为世界坐标系，所以R是单位矩阵。
        cv::Mat P1(3,4,CV_32F, cv::Scalar(0));
        K.copyTo(P1.rowRange(0,3).colRange(0,3));
        // 第一个相机的光心在世界坐标系下的坐标，原点。
        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

        // 步骤2 得到第二个相机的相机矩阵，P2=K[R|t]。
        cv::Mat P2(3,4,CV_32F);
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        P2 = K*P2;

        // 第二个相机光心在世界坐标下的坐标。
        cv::Mat O2 = -R.t()*t;


        int nGood = 0;

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            // 不好的匹配点，跳过。
            if(!vbMatchesInliers[i])
                continue;

            // kp1和kp2是匹配特征点。
            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
            cv::Mat p3dC1;

            // 步骤３　三角化配对的特征点，恢复3D坐标p3dC1。
            Triangulate(kp1, kp2, P1, P2, p3dC1);

            //　isfinite判断元素是否有界，有界返回true。
        	if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)) )
            {
                // 无界，匹配点不好。
                vbGood[vMatches12[i].first] = false;
                continue;
            }

            // 步骤４　计算视差角的余弦。
            // 即3D点与两相机光心连线的夹角的余弦。
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = cv::norm(normal2);

            float cosParallax = normal1.dot(normal2)/(dist1*dist2);

            // 步骤５　判断3D点是否在两个摄像头前方。

            // 步骤5.1　3D点深度为负，在第一个摄像头后方，剔除。
            if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
                continue;

            // 转换为第二个相机坐标系下的3D坐标。
            cv::Mat p3dC2 = R*p3dC1+t;

            // 3D点深度为负，在第二个相机后方，剔除。
            if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
                continue;

            // 步骤6.1　计算重投影误差。

            // 计算3D点在第一个图像上的重投影误差。
            float im1x, im1y;
            float invZ1 = 1.0/p3dC1.at<float>(2);
            im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
            im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

            float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

            // 步骤6.1　在第一帧图像重投影误差太大，剔除(小视角情况)。
            if(squareError1 > th2)
                continue;

            // 计算3D点在第二个图像上的冲投影误差。
            float im2x, im2y;
            float invZ2 = 1.0/p3dC2.at<float>(2);
            im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
            im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

            float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

            // 步骤6.2　在第二帧图像重投影误差太大，剔除。
            if(squareError2 > th2)
                continue;

            // 步骤７　统计检验通过的3D点个数，记录视差角。
            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
            nGood++;

            // 视差角不太小。
            if(cosParallax < 0.99998)
                vbGood[vMatches12[i].first] = true;
        }

        // 步骤８　得到3D点中比较大的视差角。
        if(nGood>0)
        {
            // 从小到大排序。
            sort(vCosParallax.begin(), vCosParallax.end());

            // 取较大的视差角。
            size_t idx = min(50, int(vCosParallax.size()-1));
            parallax = acos(vCosParallax[idx])*180/CV_PI;   // 转换成角度制。
        }
        else
            parallax = 0;

        return nGood;
    }



    //                          |0 -1  0|
    // E = U Sigma V'   let W = |1  0  0|
    //                          |0  0  1|
    // 得到4个解 E = [R|t]
    // R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3

    // 分解本征矩阵E。
    // E分解后可以得到四组，[R1,t], [R1,-t], [R2,t], [R2,-t]。
    // 具体理论见多视几何P175，结论8.19。
    void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
    {
        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        // 对t进行归一化，但是不决定SLAM的尺度。
        u.col(2).copyTo(t);
        t = t/cv::norm(t);

        cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
        W.at<float>(0,1) = -1;
        W.at<float>(1,0) = 1;
        W.at<float>(2,2) = 1;

        R1 = u*W*vt;
        // 旋转矩阵，行列式为１。
        if(cv::determinant(R1) < 0)
            R1 = -R1;

        R2 = u*W.t()*vt;
        if(cv::determinant(R2) < 0)
            R2 = -R2;
    }



}   // namespace ORB_SLAM2




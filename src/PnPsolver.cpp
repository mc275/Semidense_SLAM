
//这里的pnp求解用的是EPnP的算法。
// 参考论文：EPnP:An Accurate O(n) Solution to the PnP problem
// https://en.wikipedia.org/wiki/Perspective-n-Point
// http://docs.ros.org/fuerte/api/re_vision/html/classepnp.html
// 如果不理解，可以看看中文的："摄像机位姿的高精度快速求解" "摄像头位姿的加权线性算法"

// PnP求解：已知世界坐标系下的3D点与图像坐标系对应的2D点，求解相机的外参(R t)，即从世界坐标系到相机坐标系的变换。
// 而EPnP的思想是：
// 将世界坐标系所有的3D点坐标用以四个虚拟的控制点为基底的欧式空间来表示，
// 根据空间点和控制点的位置关系以及空间点图像，求解控制点在摄像机坐标系下的坐标，
// 根据世界坐标系下的四个控制点与相机坐标系下对应的四个控制点（与世界坐标系下四个控制点有相同尺度）即可恢复出(R t)


// EPnP的实现
// 见论文P5　3.2节。
//                                   |x|
//   |u|   |fx r  u0||r11 r12 r13 t1||y|
// s |v| = |0  fy v0||r21 r22 r23 t2||z|
//   |1|   |0  0  1 ||r32 r32 r33 t3||1|

// step1:用四个控制点来表达所有的3D点[x,y,z,1].t
// p_w = sigma(alphas_j * pctrl_w_j), j从0到4
// p_c = sigma(alphas_j * pctrl_c_j), j从0到4
// sigma(alphas_j) = 1,  j从0到4

// step2:根据针孔投影模型，相机坐标系投影到像素坐标下。
// s * U = K*p_c = K * sigma(alphas_j * pctrl_c_j), j从0到4

// step3:将step2的式子展开,利用第三行消去s
// sigma(alphas_j * fx * yctrl_c_j) + alphas_j * (cx-u)*zctrl_c_j = 0
// sigma(alphas_j * fy * yctrl_c_j) + alphas_j * (cy-v)*zctrl_c_j = 0

// step4:将step3中的12未知参数（4个控制点的3维参考点坐标）提成列向量
// Mx = 0,计算得到初始的解x后可以用Gauss-Newton来提纯得到四个相机坐标系的控制点

// step5:根据得到的p_w和对应的p_c，最小化重投影误差即可求解出R t。



#include "PnPsolver.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{

    // pcs表示3D点在相机坐标系下的坐标。
    // pws表示3D点在世界坐标系下的坐标。
    // us 表示3D点对应的2D点坐标。
    // alphas 为以４个虚拟控制点为基底，表示3D点坐标时的系数。
    // 构造函数，初始化特征点和3D坐标容器。
    PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches):
        pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
        mnIterations(0), mnBestInliers(0), N(0)
    {
        // 根据点的数量初始化容器大小。
        mvpMapPointMatches = vpMapPointMatches;
        mvP2D.reserve(F.mvpMapPoints.size());
        mvSigma2.reserve(F.mvpMapPoints.size());
        mvP3Dw.reserve(F.mvpMapPoints.size());
        mvKeyPointIndices.reserve(F.mvpMapPoints.size());
        mvAllIndices.reserve(F.mvpMapPoints.size());

        int idx = 0;
        for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
        {
            // 获取一个MapPoints。
            MapPoint *pMP = vpMapPointMatches[i];

            if(pMP)
            {
                if(!pMP->isBad())
                {
                    // 获得二维特征点。
                    const cv::KeyPoint &kp = F.mvKeysUn[i];

                    mvP2D.push_back(kp.pt);
                    mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);     // 记录特征点提取的金字塔层数。

                    cv::Mat Pos = pMP->GetWorldPos();   // 地图点云的世界坐标。
                    mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

                    mvKeyPointIndices.push_back(i);     // 记录选中特征点在原始特征点容器中的索引号，连续的。
                    mvAllIndices.push_back(idx);        // 记录选中特征点的索引，连续的。
                    
                    idx++;
                }
            }
        } 

        // 设置相机标定参数。
        fu = F.fx;
        fv = F.fy;
        uc = F.cx;
        vc = F.cy;

        SetRansacParameters();

    }

    //　析构函数。
    PnPsolver::~PnPsolver()
    {
        // 释放动态分配的数组。
        delete [] pws;
        delete [] us;
        delete [] alphas;
        delete [] pcs;
    }



    // 设置RANSAC迭代的参数。
    void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
    {
        mRansacProb = probability;
        mRansacMinInliers = minInliers;
        mRansacMaxIts = maxIterations;
        mRansacEpsilon = epsilon;
        mRansacMinSet = minSet;

        N = mvP2D.size();       // 所有二维特征点的个数。

        mvbInliersi.resize(N);  // 记录每次迭代的每对特征点的内点情况。

        int nMinInliers = N*mRansacEpsilon;     // RANSAC的残差。
        if(nMinInliers < mRansacMinInliers)
            nMinInliers = mRansacMinInliers;
        if(nMinInliers < minSet)
            nMinInliers = minSet;
        mRansacMinInliers = nMinInliers;

        if(mRansacEpsilon < (float)mRansacMinInliers/N)
            mRansacEpsilon = (float)mRansacMinInliers/N;

        int nIterations;

        // 根据残差来计算RANSAC需要迭代的次数。
        if(mRansacMinInliers == N)
            nIterations = 1;
        else
            nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

        mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

        mvMaxError.resize(mvSigma2.size());     // 图像提取特征的时候尺度层数
        for(size_t i=0; i<mvSigma2.size(); i++) // 不同尺度，设置不同的最大偏差。
            mvMaxError[i] = mvSigma2[i]*th2;

    }



    // 
    cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
    {
        bool bFlag;
        return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
    }

    //　RANSAC迭代利用EPnP求解相机位姿R,t。主函数，完成EPnP。 
    cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
    {
        bNoMore = false;
        vbInliers.clear();
        nInliers = 0;

        // mRansacMinSet为每次RANSAC需要的抽样点数，默认为4组3D-2D对应点。
        set_maximum_number_of_correspondences(mRansacMinSet);

        // N为所有2D点的个数，mRansacMinInliers为RANSAC迭代过程最少的inliers数阈值。
        if(N < mRansacMinInliers)
        {
            bNoMore = true;
            return cv::Mat();
        }

        //　mvAllIndices为所有参与PnP的2D点的索引。
        //　vAvailableIndices为每次从mvAllIndices中随机挑选的mRansacMinSet组3D-2D对应点，进行一次RANSAC迭代。
        vector<size_t> vAvailableIndices;

        int nCurrentIterations = 0;
        while(mnIterations < mRansacMaxIts || nCurrentIterations < nIterations)
        {
            nCurrentIterations++;
            mnIterations++;
            reset_correspondences();

            vAvailableIndices = mvAllIndices;

            // 随机产生RANSAC迭代需要的3D-2D对应点。
            for(short i=0; i<mRansacMinSet; i++)
            {
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

                // 随机产生的1对3D-2D点的索引。
                int idx = vAvailableIndices[randi]; 

                // 将对应的2D点和3D点压入pws和us中。
                add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);

                // 在vAvailableIndices中剔除选中的特征点对，避免重复选择3D-2D特征点对。
                vAvailableIndices[idx] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }

            //　计算相机位姿。
            compute_pose(mRi, mti);

            //　利用求得的R,t，检测所有的特征点对中哪些是内点。
            CheckInliers();

            // 检测到的内点数量与RANSAC的阈值进行比较。
            if(mnInliersi >= mRansacMinInliers)
            {
                // 大于阈值，保存。
                if(mnInliersi > mnBestInliers)
                {
                    mvbBestInliers = mvbInliersi;
                    mnBestInliers = mnInliersi;

                    cv::Mat Rcw(3,3,CV_64F, mRi);
                    cv::Mat tcw(3,1,CV_64F, mti);
                    Rcw.convertTo(Rcw,CV_32F);
                    tcw.convertTo(tcw,CV_32F);
                    mBestTcw = cv::Mat::eye(4,4,CV_32F);
                    Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                    tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
                }

                // 优化
                if(Refine())
                {
                    nInliers = mnRefinedInliers;
                    vbInliers = vector<bool>(mvpMapPointMatches.size(),false);

                    for(int i=0; i<N; i++)
                    {
                        // 设置所有地图点云的内点情况。
                        if(mvbRefinedInliers[i])
                            vbInliers[mvKeyPointIndices[i]] = true;
                    }
                    return mRefinedTcw.clone();
                }
                
            }
        }

        // 迭代次数大于mRansacMaxIts。
        if(mnIterations >= mRansacMaxIts)
        {
            bNoMore = true;
            if(mnBestInliers >= mRansacMinInliers)
            {
                nInliers = mnBestInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
                for(int i=0; i<N; i++)
                {
                    if(mvbBestInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mBestTcw.clone();

            }
        }

        return cv::Mat();

    }

    // 
    bool PnPsolver::Refine()
    {
        vector<int> vIndices;
        vIndices.reserve(mvbBestInliers.size());

        for(size_t i=0; i<mvbBestInliers.size();i++)
        {
            // 记录内点的标号。
            if(mvbBestInliers[i])
            {
                vIndices.push_back(i);
            }
        }

        // 设置每次RANSAC迭代需要的特征点对数
        set_maximum_number_of_correspondences(vIndices.size());

        reset_correspondences();

        // 对所有的3D-2D内点再进行一次EPnP。
        for(size_t i=0; i<vIndices.size(); i++)
        {
            int idx = vIndices[i];
            add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);
        }

        // 利用之前选好的所有内点，再次EPnP求解R,t。
        compute_pose(mRi, mti);

        // 根据新求解的R,t检验特征点是否是内点，进行内点提纯。相当于RANSAC求解好模型后，用所有点再次求解，进行优化。
        CheckInliers();

        mnRefinedInliers = mnInliersi;
        mvbRefinedInliers = mvbInliersi;

        if(mnInliersi>mRansacMinInliers)
        {
            cv::Mat Rcw(3,3,CV_64F, mRi);
            cv::Mat tcw(3,1,CV_64F, mti);
            Rcw.convertTo(Rcw,CV_32F);
            tcw.convertTo(tcw,CV_32F);
            mRefinedTcw = cv::Mat::eye(4,4, CV_32F);
            Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
            return true;
        }

        return false;

    }



    // 利用计算得到的R,t检查哪些3D-2D属于内点。
    void PnPsolver::CheckInliers()
    {
        mnInliersi = 0;

        // 遍历所有3D-2D匹配点。
        for(int i=0; i<N; i++)
        {
            cv::Point3f P3Dw = mvP3Dw[i];
            cv::Point2f P2D = mvP2D[i];

            // 将世界坐标系下的P3Dw投影到相机坐标系下P3Dc。
            float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
            float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
            float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

            // 讲相机坐标系下的P3Dc投影到像素坐标系下。
            double ue = uc + fu*Xc*invZc;
            double ve = vc + fv*Yc*invZc;

            // 计算投影残差。
            float distX = P2D.x - ue;
            float distY = P2D.y - ve;

            float error2 = distX*distX+distY*distY;

            // 误差小于设定阈值。
            if(error2<mvMaxError[i])
            {
                mvbInliersi[i] = true;
                mnInliersi++;
            }
            else
            {
                mvbInliersi[i] = false;
            }
        }

    }



    // 设置RANSAC每次迭代PnP求解时3D-2D点最大匹配数。
    // 决定了pws us alphas pcs容器大小。 
    void PnPsolver::set_maximum_number_of_correspondences(int n)
    {
        // 之前的maximum_number_of_correspondences过小，重新初始化pws us alphas pcs的大小。
        if(maximum_number_of_correspondences < n)
        {
            if(pws != 0)
                delete [] pws;
            if(us != 0)
                delete [] us;
            if(alphas != 0)
                delete [] alphas;
            if(pcs != 0)
                delete [] pcs;
        
            maximum_number_of_correspondences = n;
            pws = new double[3 * maximum_number_of_correspondences];        // 每个世界坐标3D点有(X Y Z)三个坐标。
            us  = new double[2 * maximum_number_of_correspondences];        // 每个2D点有(U V)两个坐标。
            alphas = new double[4 * maximum_number_of_correspondences];     // 3D坐标由4个控制点表示，四个系数。
            pcs = new double[3 * maximum_number_of_correspondences];        // 每个相机坐标3D点有(X Y Z)三个坐标。
        }

    }



    // 重置已加入PnP的3D-2D匹配点对数。
    void PnPsolver::reset_correspondences(void)
    {
        // 当前加入EPnP求解的3D-2D匹配点对数。
        number_of_correspondences = 0;
    }



    // 将一对3D-2D点加入到加入到EPnP的求解
    void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
    {
        // 3D点。
        pws[ 3*number_of_correspondences] = X;
        pws[ 3*number_of_correspondences+1] = Y;
        pws[ 3*number_of_correspondences+2] = Z;

        // 2D点。
        us[ 2*number_of_correspondences] = u;
        us[ 2*number_of_correspondences+1] = v;

        // 进行EPnP求解的3D-2D配对点数量+1。
        number_of_correspondences++;
        
    }



    // 选择用于表示所有3D点的４个控制点。
    void PnPsolver::choose_control_points(void)
    {
        // 步骤1 第一个控制点，参与PnP计算的所有点的几何中心。
        cws[0][0] = cws[0][1] = cws[0][2]=0;      // X Y Z初始化

        for(int i=0; i<number_of_correspondences; i++)   // 遍历3D点
            for(int j=0; j<3; j++)                      // 遍历3D点的X Y Z坐标
                cws[0][j] += pws[3*i+j];

        for(int j=0; j<3; j++)
            cws[0][j] /= number_of_correspondences; 

        // 步骤2 计算其他三个控制点，通过PCA分解得到。
        //       将所有3D点写成矩阵(number_of_correspondences * 3)。
        CvMat *PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

        double pw0tpw0[3*3], dc[3], uct[3*3];
        CvMat PW0tPW0 = cvMat(3,3,CV_64F,pw0tpw0);
        CvMat DC = cvMat(3,1,CV_64F, dc);
        CvMat UCt = cvMat(3,3,CV_64F, uct);
        
        // 步骤2.1　将pws中的参考3D点减去第一个控制点的坐标(相当于第一个点是原点)， 存入PW0。
        for(int i=0; i<number_of_correspondences; i++)
            for(int j=0; j<3; j++)
                PW0->data.db[3*i+j] = pws[3*i+j] - cws[0][j];

        // 步骤2.2　利用SVD分解PW0'PW0可以获得主分量。
        // 类似于齐次线性最小二乘求解。
        cvMulTransposed(PW0, &PW0tPW0, 1);
        cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

        cvReleaseMat(&PW0);

        // 步骤2.3　得到C1, C2, C3三个3D控制点，最后加上减去的偏移量。
        for(int i=1; i<4; i++)
        {
            double k = sqrt(dc[i-1]/number_of_correspondences);
            for(int j=0; j<3; j++)
                cws[i][j] = cws[0][j] + k*uct[3*(i-1)+j];
        }

    }



    // 求解4个控制点的系数alphas。
    // (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1)，a1 = 1-a2-a3-a4。
    void PnPsolver::compute_barycentric_coordinates(void)
    {
        double cc[3*3], cc_inv[3*3];
        CvMat CC = cvMat(3, 3, CV_64F, cc);
        CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

        // 第一个控制点在质心位置，后三个控制点减去第一个控制点(以第一个控制点为原点)。
        // 步骤１　减去质心后得到x y z轴。
        // 
        // cws的排列 |cws1_x cws1_y cws1_z|  ---> |cws1|
        //           |cws2_x cws2_y cws2_z|       |cws2|
        //           |cws3_x cws3_y cws3_z|       |cws3|
        //           |cws4_x cws4_y cws4_z|       |cws4|
        //          
        // cc的排列  |cc2_x cc3_x cc4_x|  --->|cc2 cc3 cc4|
        //           |cc2_y cc3_y cc4_y|
        //           |cc2_z cc3_z cc4_z|
        for(int i=0; i<3; i++)          // 遍历XYZ坐标。
            for(int j=1; j<4; j++)      // 遍历c2 c3 c4。
                cc[3*i+j-1] = cws[j][i] - cws[0][i];

        cvInvert(&CC, &CC_inv, CV_SVD);
        double *ci = cc_inv;
        for(int i=0; i<number_of_correspondences; i++)
        {
            double *pi = pws+3*i;   // pi指向第i个3D点首地址。
            double *a = alphas+4*i; // a指向第i个3D点的控制点系数首地址。

            // 求解pi用控制点cws表示的坐标，[a1,a2,a3,a4]
            for(int j=0; j<3; j++)
                a[1+j]= ci[3*j  ]*(pi[0]-cws[0][0])+
                        ci[3*j+1]*(pi[1]-cws[0][1])+
                        ci[3*j+2]*(pi[2]-cws[0][2]);
            a[0] = 1.0f-a[1]-a[2]-a[3];
        }

    }



    //　填充最小二乘矩阵M。
    //　对每个3D-2D参考点pi-ui
    // |fu*ai1 0    -ai1*(uc-ui), fu*ai2  0    -ai2*(uc-ui), fu*ai3 0   -ai3*(uc-ui), fu*ai4 0   -ai4*(uc-ui)|
    // |0   fv*ai1  -ai1*(vc-vi), 0    fv*ai2  -ai2*(vc-vi), 0   fv*ai3 -ai3*(vc-vi), 0   fv*ai4 -ai4*(vc-vi)|
    void PnPsolver::fill_M(CvMat *M, 
                const int row, const double *as, const double u, const double v)
    {
        double *M1 = M->data.db + row*12;   // 第i个3D-2D点的M矩阵的第1行。
        double *M2 = M1 + 12;               // 第i个3D-2D点的M矩阵的第2行。

        for(int i=0; i<4; i++)
        {
            // 第一行的4组元素，每组3个。i相当于上面公式的1,2,3,4
            M1[3*i  ] = as[i]*fu;
            M1[3*i+1] = 0.0;
            M1[3*i+2] = as[i]*(uc-u);

            // 第二行。
            M2[3*i  ] = 0.0;
            M2[3*i+1] = as[i]*fv;
            M2[3*i+2] = as[i]*(vc-v);
        }

    }



    // 通过betas和ut合成相机坐标系下的控制点。
    // 每个控制点在相机坐标系下都表示成特征向量乘以beta的形式，EPnP论文公式16。
    void PnPsolver::compute_ccs(const double *betas, const double *ut)
    {
        // 初始化相机坐标系下的控制点坐标。
        for(int i=0; i<4; i++)
            ccs[i][0] = ccs[i][1]=ccs[i][2] = 0.0f;

        for(int i=0; i<4; i++)
        {
            const double *v = ut + 12*(11-i);
            
            for(int j=0; j<4; j++)              // 遍历4个控制点。
                for(int k=0; k<3; k++)          // 遍历XYZ坐标。
                    ccs[j][k] += betas[i]*v[3*j+k];
        }

    }



    // 用4个控制点作为基底表示相机坐标系下的3D点坐标，pcs保存所有3D点的相机坐标系坐标。
    void PnPsolver::compute_pcs(void)
    {
        for(int i=0; i<number_of_correspondences; i++)
        {
            double *a = alphas+4*i;     // 第i个3D点的4个系数的首地址。
            double *pc = pcs +3*i;      // 第i个3D点的坐标的首地址。

            for(int j=0; j<3; j++)
                pc[j] = a[0]*ccs[0][j]+a[1]*ccs[1][j]+a[2]*ccs[2][j]+a[3]*ccs[3][j];
        }

    }



    //  计算相机坐标系下的控制点坐标(只计算了betas和v, compute_ccs进行坐标合成)。
    //　计算相机位姿R,t，并选出最优。
    //　返回最小的重投影误差。
    double PnPsolver::compute_pose(double R[3][3], double t[3])
    {
        // 步骤１　得到EPnP算法中的４个控制点。
        choose_control_points();

        // 步骤２　计算世界坐标系下每个3D点用4个控制点表示的坐标。
        compute_barycentric_coordinates();

        // 步骤３　构造M矩阵。
        CvMat *M = cvCreateMat(2*number_of_correspondences, 12, CV_64F);

        for(int i=0; i<number_of_correspondences; i++)
            fill_M(M, 2*i, alphas+4*i, us[2*i], us[2*i+1]);

        double mtm[12*12], d[12], ut[12*12];
        CvMat MtM = cvMat(12, 12, CV_64F, mtm);
        CvMat D   = cvMat(12, 1,  CV_64F, d  );
        CvMat Ut  = cvMat(12, 12, CV_64F, ut );

        // 步骤3 求解Mx = 0。
        // 求解M'M比M运算速度更快。
        cvMulTransposed(M, &MtM, 1);    // MtM = M'M。
        cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);  
        cvReleaseMat(&M);

        double l_6x10[6*10], rho[6];
        CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
        CvMat Rho = cvMat(6, 1, CV_64F, rho);

        compute_L_6x10(ut, l_6x10);
        compute_rho(rho);

        double Betas[4][4], rep_errors[4];
        double Rs[4][3][3], ts[4][3];

        // 求部分betas，通过优化得出全部betas。
        
        // 假设M'M的零空间的维度是1的公式计算betas。
        find_betas_approx_1(&L_6x10, &Rho, Betas[1]);                       // 公式10
        gauss_newton(&L_6x10, &Rho, Betas[1]);                              // 公式15
        rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);        // 计算当前情况下求取R,t后的重投影误差。

        // 假设M'M的零空间的维度是2的公式计算betas。
        find_betas_approx_2(&L_6x10, &Rho, Betas[2]);                       // 公式11
        gauss_newton(&L_6x10, &Rho, Betas[2]);                              // 公式15
        rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);        // 计算重投影误差。

        // 假设维度3的公式计算。
        find_betas_approx_3(&L_6x10, &Rho, Betas[3]);                       // 和公式11相似，只不过L是可逆矩阵，不需要伪逆。
        gauss_newton(&L_6x10, &Rho, Betas[3]);                              // 公式15。
        rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);        // 计算重投影误差。

        // 需找最小的冲投影误差对应的解。
        int N=1;
        if(rep_errors[2] < rep_errors[1])
            N=2;
        if(rep_errors[3] < rep_errors[N])
            N=3;

        // 保存最小重投影误差对应的R和t。
        copy_R_and_t(Rs[N], ts[N], R, t);

        return rep_errors[N];       

    }



    // 复制矩阵。
    void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3], double R_dst[3][3], double t_dst[3])
    {
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
                R_dst[i][j] = R_src[i][j];
            
            t_dst[i] = t_src[i];
        }

    }



    // 计算3D坐标之间的距离的平方
    double PnPsolver::dist2(const double *p1, const double *p2)
    {
        return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]);
    }

    // 两向量点乘。
    double PnPsolver::dot(const double *v1, const double *v2)
    {
        return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
    }



    // 计算重投影误差。
    double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
    {
        double sum2 = 0.0;
        
        for(int i=0; i < number_of_correspondences; i++)    
        {
            double *pw = pws+3*i;
            double Xc = dot(R[0], pw)+t[0];
            double Yc = dot(R[1], pw)+t[1];
            double inv_Zc = 1.0/(dot(R[2], pw)+t[2]);
            double ue = uc+fu*Xc*inv_Zc;
            double ve = vc+fv*Yc*inv_Zc;
            double u = us[2*i], v=us[2*i+1];

            sum2 += sqrt((u-ue)*(u-ue)+(v-ve)*(v-ve));
        }

        return sum2 / number_of_correspondences;

    }



    // 根据世界坐标系下的4个控制点坐标与相机坐标系下(相同尺度)的4个控制点坐标，求R,t。3D-3D问题
    void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
    {
        double pc0[3], pw0[3];

        pc0[0] = pc0[1] = pc0[2] = 0.0;
        pw0[0] = pw0[1] = pw0[2] = 0.0;

        for(int i=0; i<number_of_correspondences; i++)
        {
            const double *pc = pcs + 3*i;
            const double *pw = pws + 3*i;

            for(int j=0; j<3; j++)
            {
                pc0[j] += pc[j];
                pw0[j] += pw[j];
            }
        }

        for(int j=0; j<3; j++)
        {
            // 质心坐标。
            pc0[j] /= number_of_correspondences;
            pw0[j] /= number_of_correspondences;
        }

        double abt[3*3], abt_d[3], abt_u[3*3], abt_v[3*3];

        CvMat ABt   = cvMat(3, 3, CV_64F, abt);
        CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
        CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
        CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

        cvSetZero(&ABt);

        for(int i=0; i<number_of_correspondences; i++)
        {
            double *pc = pcs+3*i;
            double *pw = pws+3*i;

            // sum(qi*qi'.t())，见3D-3D。
            for(int j=0; j<3; j++)      // 遍历abt矩阵的每一行
            {
                // 操作第j行上的每个元素。
                abt[3*j  ] += (pc[j]-pc0[j])*(pw[0]-pw0[0]);
                abt[3*j+1] += (pc[j]-pc0[j])*(pw[1]-pw0[1]);
                abt[3*j+2] += (pc[j]-pc0[j])*(pw[2]-pw0[2]);
            }
        }

        cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

        //　旋转矩阵 R=UV.t
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                R[i][j] = dot(abt_u+3*i, abt_v+3*j);

        // 矩阵R的行列式值。
        const double det =
            R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
            R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

        // 旋转矩阵 |R| = 1
        if(det < 0)
        {
            R[2][0] = -R[2][0];
            R[2][1] = -R[2][1];
            R[2][2] = -R[2][2];
        }

        // t = pc0 - R*pw0。
        t[0] = pc0[0] - dot(R[0], pw0);
        t[1] = pc0[1] - dot(R[1], pw0);
        t[2] = pc0[2] - dot(R[2], pw0);

    }



    // 输出测试代码。
    void PnPsolver::print_pose(const double R[3][3], const double t[3])
    {
        cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
        cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
        cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
    }



    // 解决深度为负的符号问题
    void PnPsolver::solve_for_sign(void)
    {
        // 深度为负。
        if(pcs[2] < 0.0)
        {
            // 保证alpha不变，同时变号。
            for(int i=0; i<4; i++)
                for(int j=0; j<3; j++)
                    ccs[i][j] = -ccs[i][j];

            for(int i=0; i<number_of_correspondences; i++)
            {
                pcs[3*i  ] = -pcs[3*i  ];
                pcs[3*i+1] = -pcs[3*i+1];
                pcs[3*i+2] = -pcs[3*i+2];
            }
        }
    }



    // 计算相机R,t, 返回当前状态下的重投影误差。
    double PnPsolver::compute_R_and_t(const double *ut, const double *betas, 
                    double R[3][3], double t[3])
    {
        // 计算控制点在相机坐标下的坐标。
        compute_ccs(betas, ut);

        // 利用控制点计算所有3D点在相机坐标系的坐标。
        compute_pcs();

        // 解决坐标符号问题，深度<0。
        solve_for_sign();

        // 通过3D-3D计算相机位姿R,t。
        estimate_R_and_t(R, t);

        // 返回当前R, t的重投影误差。
        return reprojection_error(R, t);

    }



    // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_1 = [B11 B12     B13         B14]  
    void PnPsolver::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas)
    {
        double l_6x4[6*4], b4[4];
        CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
        CvMat B4    = cvMat(4, 1, CV_64F, b4);

        for(int i=0; i<6; i++)
        {
            cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
            cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
        }

        cvSolve(&L_6x4, Rho, &B4, CV_SVD);

        if(b4[0] < 0)
        {
            betas[0] = sqrt(-b4[0]);
            betas[1] = -b4[1] / betas[0];
            betas[2] = -b4[2] / betas[0];
            betas[3] = -b4[3] / betas[0];
        }

        else
        {
            betas[0] = sqrt(b4[0]);
            betas[1] = b4[1] / betas[0];
            betas[2] = b4[2] / betas[0];
            betas[3] = b4[3] / betas[0];
        }

    }



    // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_2 = [B11 B12 B22                            ]
    void PnPsolver::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho, double *betas)
    {

        double l_6x3[6 * 3], b3[3];
        CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
        CvMat B3     = cvMat(3, 1, CV_64F, b3);

        for(int i = 0; i < 6; i++) 
        {
            cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
        }

        cvSolve(&L_6x3, Rho, &B3, CV_SVD);

        if(b3[0] < 0)
        {
            betas[0] = sqrt(-b3[0]);
            betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
        }
        else
        {
            betas[0] = sqrt(b3[0]);
            betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
        }

        if(b3[1] < 0 )
            betas[0] = -betas[0];

        betas[2] = 0.0;
        betas[3] = 0.0;
    }



    // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_3 = [B11 B12 B22 B13 B23                    ]
    void PnPsolver::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas)
    {
        double l_6x5[6 * 5], b5[5];
        CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
        CvMat B5    = cvMat(5, 1, CV_64F, b5);

        for(int i = 0; i < 6; i++) 
        {
            cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
            cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
            cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
            cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
            cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
        }
        
        cvSolve(&L_6x5, Rho, &B5, CV_SVD);

        if (b5[0] < 0)
        {
            betas[0] = sqrt(-b5[0]);
            betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;

        } 
        
        else 
        {
            betas[0] = sqrt(b5[0]);
            betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
        }
        
        if (b5[1] < 0) 
            betas[0] = -betas[0];
        
        betas[2] = b5[3] / betas[0];
        betas[3] = 0.0;
    }



    // 计算矩阵L。
    void PnPsolver::compute_L_6x10(const double *ut, double *l_6x10)
    {
        const double *v[4]; 

        // ut的最后4行。
        v[0] = ut + 12*11;
        v[1] = ut + 12*10;
        v[2] = ut + 12*9;
        v[3] = ut + 12*8;

        double dv[4][6][3];

        for(int i=0; i<4; i++)
        {
            int a=0, b=1;
            for(int j=0; j<6; j++)
            {
                dv[i][j][0] = v[i][3*a  ] - v[i][3*b  ];
                dv[i][j][1] = v[i][3*a+1] - v[i][3*b+1];
                dv[i][j][2] = v[i][3*a+2] - v[i][3*b+2];

                b++;
                if(b>3)
                {
                    a++;
                    b = a+1;
                }
            }
        }

        for(int i=0; i<6; i++)
        {
            double *row = l_6x10 + 10*i;

            row[0] =        dot(dv[0][i], dv[0][i]);
            row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
            row[2] =        dot(dv[1][i], dv[1][i]);
            row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
            row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
            row[5] =        dot(dv[2][i], dv[2][i]);
            row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
            row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
            row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
            row[9] =        dot(dv[3][i], dv[3][i]);
        }

    }



    // 计算两个控制点任意两点间的距离。
    void PnPsolver::compute_rho(double *rho)
    {
        rho[0] = dist2(cws[0], cws[1]);
        rho[1] = dist2(cws[0], cws[2]);
        rho[2] = dist2(cws[0], cws[3]);
        rho[3] = dist2(cws[1], cws[2]);
        rho[4] = dist2(cws[1], cws[3]);
        rho[5] = dist2(cws[2], cws[3]);

    }



    void PnPsolver::compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho, 
                        double betas[4], CvMat *A, CvMat *b)
    {
        for(int i=0; i<6; i++)
        {
            const double *rowL = l_6x10 + i*10;
            double *rowA = A->data.db + i*4;

            rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
            rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
            rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
            rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

            cvmSet(b, i, 0, rho[i] -
                (
                    rowL[0] * betas[0] * betas[0] +
                    rowL[1] * betas[0] * betas[1] +
                    rowL[2] * betas[1] * betas[1] +
                    rowL[3] * betas[0] * betas[2] +
                    rowL[4] * betas[1] * betas[2] +
                    rowL[5] * betas[2] * betas[2] +
                    rowL[6] * betas[0] * betas[3] +
                    rowL[7] * betas[1] * betas[3] +
                    rowL[8] * betas[2] * betas[3] +
                    rowL[9] * betas[3] * betas[3]
                ) );
        }

    }



    // 用G-N方法优化betas。
    void PnPsolver::gauss_newton(const CvMat *L_6x10, const CvMat *Rho, double betas[4])
    {
        const int iterations_number = 5;

        double a[6*4], b[6], x[4];
        CvMat A = cvMat(6, 4, CV_64F, a);
        CvMat B = cvMat(6, 1, CV_64F, b);
        CvMat X = cvMat(4, 1, CV_64F, x);

        for(int k=0; k<iterations_number; k++)
        {
            // 计算G-N方法的A和B矩阵(J.t*J, -J.t )
            compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);

            // 求解，得到的是增量。
            qr_solve(&A, &B, &X);

            for(int i=0; i<4; i++)
                betas[i] += x[i];
        }
    }
    
    
    
    // G-N求解。
    void PnPsolver::qr_solve(CvMat * A, CvMat * b, CvMat * X)
    {
        static int max_nr = 0;
        static double * A1, * A2;

        // 统计A的行列。
        const int nr = A->rows;
        const int nc = A->cols;

        if (max_nr != 0 && max_nr < nr) 
        {
            delete [] A1;
            delete [] A2;
        }

        if (max_nr < nr) 
        {
            max_nr = nr;
            A1 = new double[nr];
            A2 = new double[nr];
        }

        double *pA = A->data.db, *ppAkk = pA;

        for(int k = 0; k < nc; k++) 
        {
            double * ppAik = ppAkk, eta = fabs(*ppAik);
            
            for(int i = k + 1; i < nr; i++) 
            {
                double elt = fabs(*ppAik);
                
                if (eta < elt) 
                    eta = elt;
                
                ppAik += nc;
            }

            if (eta == 0) 
            {
              A1[k] = A2[k] = 0.0;
              cerr << "God damnit, A is singular, this shouldn't happen." << endl;
              return;
            }

            else 
            {
                double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
                
                for(int i = k; i < nr; i++) 
                {
                    *ppAik *= inv_eta;
                    sum += *ppAik * *ppAik;
                    ppAik += nc;
                }
                
                double sigma = sqrt(sum);
                
                if (*ppAkk < 0)
                    sigma = -sigma;
                
                *ppAkk += sigma;
                A1[k] = sigma * *ppAkk;
                A2[k] = -eta * sigma;
                
                for(int j = k + 1; j < nc; j++) 
                {
                    double * ppAik = ppAkk, sum = 0;
                    
                    for(int i = k; i < nr; i++) 
                    {
                        sum += *ppAik * ppAik[j - k];
                        ppAik += nc;
                    }
                    
                    double tau = sum / A1[k];
                    ppAik = ppAkk;
                    
                    for(int i = k; i < nr; i++) 
                    {
                        ppAik[j - k] -= tau * *ppAik;
                        ppAik += nc;
                    }
                }
            }
            ppAkk += nc + 1;
        }

        // b <- Qt b
        double * ppAjj = pA, * pb = b->data.db;
        
        for(int j = 0; j < nc; j++) 
        {
            double * ppAij = ppAjj, tau = 0;
            
            for(int i = j; i < nr; i++)	
            {
                tau += *ppAij * pb[i];
                ppAij += nc;
            }

            tau /= A1[j];
            ppAij = ppAjj;
            
            for(int i = j; i < nr; i++) 
            {
                pb[i] -= tau * *ppAij;
                ppAij += nc;
            }
            
            ppAjj += nc + 1;
        }

        // X = R-1 b
        double * pX = X->data.db;
        pX[nc - 1] = pb[nc - 1] / A2[nc - 1];

        for(int i = nc - 2; i >= 0; i--) 
        {
            double * ppAij = pA + i * nc + (i + 1), sum = 0;

            for(int j = i + 1; j < nc; j++) 
            {
                sum += *ppAij * pX[j];
                ppAij++;
            }

            pX[i] = (pb[i] - sum) / A2[i];
        }
    }



    //　测试代码，与真实值进行比较。
    void PnPsolver::relative_error(double & rot_err, double & transl_err,
            const double Rtrue[3][3], const double ttrue[3],
            const double Rest[3][3],  const double test[3])
    {
        double qtrue[4], qest[4];

        mat_to_quat(Rtrue, qtrue);
        mat_to_quat(Rest, qest);

        double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
                    (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
                    (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
                    (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
            sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

        double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
                     (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
                     (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
                     (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
            sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

        rot_err = min(rot_err1, rot_err2);

        transl_err =
            sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
            (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
            (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
            sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
    }

    // 旋转矩阵转换为4元数，测试代码。
    void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
    {
        double tr = R[0][0] + R[1][1] + R[2][2];
        double n4;

        if (tr > 0.0f) 
        {
            q[0] = R[1][2] - R[2][1];
            q[1] = R[2][0] - R[0][2];
            q[2] = R[0][1] - R[1][0];
            q[3] = tr + 1.0f;
            n4 = q[3];
        }
        else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) 
        {
            q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
            q[1] = R[1][0] + R[0][1];
            q[2] = R[2][0] + R[0][2];
            q[3] = R[1][2] - R[2][1];
            n4 = q[0];
        } 
        else if (R[1][1] > R[2][2]) 
        {
            q[0] = R[1][0] + R[0][1];
            q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
            q[2] = R[2][1] + R[1][2];
            q[3] = R[2][0] - R[0][2];
            n4 = q[1];
        } 
        else 
        {
            q[0] = R[2][0] + R[0][2];
            q[1] = R[2][1] + R[1][2];
            q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
            q[3] = R[0][1] - R[1][0];
            n4 = q[2];
        }
        double scale = 0.5f / double(sqrt(n4));

        q[0] *= scale;
        q[1] *= scale;
        q[2] *= scale;
        q[3] *= scale;

    }


}   // namespace ORB_SLAM2




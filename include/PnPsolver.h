//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "MapPoint.h"
#include "Frame.h"


namespace ORB_SLAM2
{
    
    class PnPsolver
	{
        public:
            // 构造函数。
            PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches);

            // 析构函数。
            ~PnPsolver();

            // 设置RANSAC参数。
            void SetRansacParameters(double probability=0.99, int minInliers=8, int maxIterations=300, int minSet=4, float epsilon=0.4, float th2=5.991);

            // 寻找内点。
            cv::Mat find(vector<bool> &vbInliers, int &nInliers); 

            // 迭代。
            cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

        private:

            void CheckInliers();
            bool Refine();

            // 函数来源于EPnP代码。
            void set_maximum_number_of_correspondences(const int n);
            void reset_correspondences(void);
            void add_correspondence(const double X, const double Y, const double Z, const double u, const double v);
            
            // 计算位姿。
            double compute_pose(double R[3][3], double T[3]);

            // 计算相对误差。
            void relative_error(double &rot_err, double &transl_err, const double Rtrue[3][3], const double ttrue[3], const double Rest[3][3], const double test[3]);

            // 输出位姿。
            void print_pose(const double R[3][3], const double t[3]);
            
            // 计算重投影误差。
            double reprojection_error(const double R[3][3], const double t[3]);

            void choose_control_points(void);
            void compute_barycentric_coordinates(void);
            void fill_M(CvMat *M, const int row, const double * alphas, const double u, const double v);
            void compute_ccs(const double *betas, const double *ut);
            void compute_pcs(void);

            void solve_for_sign(void);

            void find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas);
            void find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho, double *betas);
            void find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho, double *betas);
            void qr_solve(CvMat *A, CvMat *b, CvMat *X);
            
            double dot(const double *v1, const double *v2);
            double dist2(const double *p1, const double *p2);

            void compute_rho(double *rho);
            void compute_L_6x10(const double *ut, double *l_6x10);
            
            // G-N求解。
            void gauss_newton(const CvMat *L_6x10, const CvMat *Rho, double current_betas[4]);
            void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho, double cb[4], CvMat *A, CvMat *b);

            // 计算R和t。
            double compute_R_and_t(const double *ut, const double *betas, double R[3][3], double t[3]);

            // 估计R和t。
            void estimate_R_and_t(double R[3][3], double t[3]);

            // 复制R和t。
            void copy_R_and_t(const double R_dst[3][3], const double t_dst[3], double R_src[3][3], double t_src[3]);

            // 矩阵变为四元数。
            void mat_to_quat(const double R[3][3], double q[4]);

            // 相机内参。
            double uc, vc, fu, fv;

            double *pws, *us, *alphas, *pcs;
            int maximum_number_of_correspondences;
            int number_of_correspondences;

            double cws[4][3], ccs[4][3];
            double cws_determinant;

            vector<MapPoint *> mvpMapPointMatches;

            // 2D点。
            vector<cv::Point2f> mvP2D;
            vector<float> mvSigma2;

            // 3D点。
            vector<cv::Point3f> mvP3Dw;

            // 帧索引。
            vector<size_t> mvKeyPointIndices;

            // 当前估计值。
            double mRi[3][3];
            double mti[3];
            cv::Mat mTcwi;
            vector<bool> mvbInliersi;
            int mnInliersi;

            // 当前RANSAC状态。
            int mnIterations;
            vector<bool> mvbBestInliers;
            int mnBestInliers;
            cv::Mat mBestTcw;

            // 优化。
            cv::Mat mRefinedTcw;
            vector<bool> mvbRefinedInliers;
            int mnRefinedInliers;

            // 一致特征点数。
            int N;

            // 随机选择索引数[0,...N-1]。
            vector<size_t> mvAllIndices;

            // RANSAC 概率。
            double mRansacProb;

            // RANSAC 最少内点。
            int mRansacMinInliers;

            // RANSAC 最大迭代次数。
            int mRansacMaxIts;

            // RANSAC 期望内点比率。
            float mRansacEpsilon;

            // RANSAC 内点比外点阈值。最大误差e=dist(P1, T12*P2)^2。
            float mRansacTh;

            // RANSAC 每次迭代最小化设置。
            int mRansacMinSet;

            // 最大误差平方，和尺度层级有关系。 最大误差 error=th*th*sigma(level)*sigma(level)。
            vector<float> mvMaxError;






    };


}   // namespace ORB_SLAM2 




#endif


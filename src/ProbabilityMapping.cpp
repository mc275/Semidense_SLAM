

#include <cmath>
#include <numeric>
#include <stdint.h>
#include <stdio.h>

#include "ProbabilityMapping.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "LocalMapping.h"

// #define InterKeyFrameChecking


template <typename T>
float bilinear(const cv::Mat& img, const float& y, const float& x)
{
    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0 + 1;

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;

    
    float interpolated =
            img.at<T>(y0 , x0 ) * x0_weight + img.at<T>(y0 , x1)* x1_weight +
            img.at<T>(y1 , x0 ) * x0_weight + img.at<T>(y1 , x1)* x1_weight +
            img.at<T>(y0 , x0 ) * y0_weight + img.at<T>(y1 , x0)* y1_weight +
            img.at<T>(y0 , x1 ) * y0_weight + img.at<T>(y1 , x1)* y1_weight ;

  return (interpolated * 0.25f);
}

namespace ORB_SLAM2 {


    // 构造函数。
    ProbabilityMapping::ProbabilityMapping(Map *pMap): 
        mpMap(pMap)
    {
        mbFinishRequested = false;
	mbFinished = true;
    }
    
    
    
    // 线程入口函数。
    void ProbabilityMapping::Run()
    {
	mbFinished = false;
	
        while(1)
        {
	  
	    if(CheckFinish())
                break;
	    
	    sleep(1);
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            SemiDenseLoop();
	    
	     
        }
        
        SetFinish();

    }
    
    
    
    // 半稠密算法实现。
    void ProbabilityMapping::SemiDenseLoop()
    {

        unique_lock<mutex> lock(mMutexSemiDense);

        // 读取地图中的关键帧。
        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        cout << "semidense_Info: vpKFs.size() --> " << vpKFs.size() << endl;
        if(vpKFs.size() < covisN +3)
            return ;

        // 遍历当前稀疏地图的关键帧。
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // 读取当前帧。
            KeyFrame *pKF = vpKFs[i];

            // 坏帧或者已经被选择。
            if(pKF->isBad() || pKF->semidense_flag_)
                continue;

            vector<KeyFrame *> closestMatches = pKF->GetBestCovisibilityKeyFrames(covisN);

            if(closestMatches.size() < covisN)
                continue;

            float max_depth;
            float min_depth;

            // 获取当前关键帧的最大、最小值和极线搜索范围。
            StereoSearchConstraints(pKF, &min_depth, &max_depth);

            cv::Mat image = pKF->GetImage();            // 666 直接返回成员变量。
	    
	    // 计算帧间的基本矩阵。
	    std::vector <cv::Mat> F;
	    F.clear();
	    for(size_t j=0; j<closestMatches.size(); j++)
	    {
	      KeyFrame* pKF2 = closestMatches[j];
	      cv::Mat F12 = ComputeFundamental(pKF,pKF2);
	      F.push_back(F12);
	    }


            // 遍历图像坐标。
            for(int img_y=2; img_y < image.rows-2; img_y++)
            {
                for(int img_x=2; img_x < image.cols-2; img_x++)
                {

                    // 像素梯度不满足阈值。
                    if(pKF->GradImg.at<float>(img_y, img_x) < lambdaG)
                        continue;

                    float pixel = image.at<uchar>(img_y,img_x);         // 666 或许应该把他弄成个cvMat

                    std::vector<depthHo> depth_ho;                      // 666
                    depth_ho.clear();

                    // 遍历当前帧cov图的相连帧。
                    for(size_t j=0; j<closestMatches.size(); j++)
                    {
                        KeyFrame *pKF2 = closestMatches[j];
			
			cv::Mat F12 = F[j];

                        // 极线搜索。
                        float best_u(0.0), best_v(0.0);
                        depthHo dh;     // 666
                        EpipolarSearch(pKF, pKF2, img_x, img_y, pixel, min_depth, max_depth, &dh, F12, best_u, best_v, pKF->GradTheta.at<float>(img_y, img_x));        // 666
                        if(dh.supported && 1/dh.depth >0.0)
                            depth_ho.push_back(dh);             // 666
                        
                    }

                    // 逆深度假设融合
                    if(depth_ho.size())
                    {
                        depthHo dh_temp;        // 666
                        InverseDepthHypothesisFusion(depth_ho, dh_temp);

			// cout << "dh_temp.supported is " << dh_temp.supported << endl;
                        if(dh_temp.supported)
                        {
                            // 用于帧内逆深度检验。
                            pKF->depth_map_.at<float>(img_y, img_x) = dh_temp.depth;
                            pKF->depth_sigma_.at<float>(img_y, img_x) = dh_temp.sigma;
                        }
                    }

                }
            }


            // 帧间你深度假设检验。
            IntraKeyFrameDepthChecking(pKF->depth_map_, pKF->depth_sigma_, pKF->GradImg);

        #ifndef InterKeyFrameChecking

            for(int img_y=2; img_y<image.rows-2; img_y++)
            {
                for(int img_x=2; img_x<image.cols-2; img_x++)
                {
                    if(pKF->depth_map_.at<float>(img_y,img_x) < 0.001 )
                        continue;

                    float inv_d = pKF->depth_map_.at<float>(img_y, img_x);
                    float Z = 1.0/inv_d;
                    float X = Z*(img_x - pKF->cx)/pKF->fx;
                    float Y = Z*(img_y - pKF->cy)/pKF->fy;

                    // 相机坐标系下的地图坐标。
                    cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y, Z, 1);
                    cv::Mat Twc = pKF->GetPoseInverse();
                    // 世界坐标系下坐标。
                    cv::Mat pos = Twc * Pc;

                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+0) = pos.at<float>(0);
                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+1) = pos.at<float>(1);
                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+2) = pos.at<float>(2);

                }
            }
            pKF->semidense_flag_ = true;        // 该帧已进行过办稠密重建。

        #endif
            
        }

        #ifdef InterKeyFrameChecking

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];

            if(pKF->isBad() || pKF->semidense_flag_) 
                continue;

            InterKeyFrameDepthChecking(pKF);

            for(int img_y=2; img_y<pKF->im_.rows-2; img_y++)
            {
                for(int img_x=2; img_x<pKF->im_.cols-2; img_x++)
                {
                    if(pKF->depth_map_.at<float>(img_y, img_x) < 0.0001)
                        continue;

                    float inv_d = pKF->depth_map_.at<float>(img_y, img_x); 
                    float Z = 1.0/inv_d;
                    float X = Z*(img_x - pKF->cx)/pKF->fx;
                    float Y = Z*(img_y - pKF->cy)/pKF->fy;

                    // 相机坐标系
                    cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y, Z, 1);
                    cv::Mat Twc = pKF->GetPoseInverse();
                    // 世界坐标系。
                    cv::Mat pos = Twc * Pc;     

                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+0) = pos.at<float>(0);
                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+1) = pos.at<float>(1);
                    pKF->SemiDensePointSets_.at<float>(img_y, 3*img_x+2) = pos.at<float>(2);
                }
            }

            pKF->semidense_flag_ = true;
        }

        #endif

    }

    
    
    // 极线搜索约束。
    void ProbabilityMapping::StereoSearchConstraints(ORB_SLAM2::KeyFrame *pKF, float *min_depth, float *max_depth)
    {
        // orb地图中降序前20的深度。
        vector<float> orb_depths = pKF->GetAllPointDepths(20);

        // 深度均值。
        float sum = accumulate(orb_depths.begin(), orb_depths.end(), 0.0);
        float mean = sum/orb_depths.size();

        vector<float> diff(orb_depths.size());

        // 使orb_depths中的对象根据迭代器索引，执行minus()操作。
        // std::bind2nd(std::minus<float>(), mean) 绑定对象mean作为minus函数的第二个参数。
        // diff = orb_depths-mean
        transform(orb_depths.begin(), orb_depths.end(), diff.begin(), bind2nd(minus<float>(), mean));
        
        // 样本方差
        // inner_product() 计算两个vector的内积。
        float variance = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)/(orb_depths.size()-1);
        // 标准差。
        float stdev = sqrt(variance);

        *max_depth = mean + 2*stdev;
        *min_depth = mean - 2*stdev;

    }
    
    
    
    // 6666
    /**
    *  极线搜索。
    *
    *  输入参数
    *    pKF1, pKF2                 两帧关键帧。
    *    img_x, img_y, pixel        pKF1中待搜索像素的坐标和灰度值。
    *    min_depth, max_depth       像素深度先验信息。
    *    th_pi                      待搜索像素梯度方向。
    *
    *  返回的指针和引用。
    *    dh                         待搜索像素的逆深度假设。
    *    best_u, best_v             待搜索像素的匹配像素坐标。                     
    **/
    void ProbabilityMapping::EpipolarSearch(KeyFrame *pKF1, KeyFrame *pKF2, const int img_x, const int img_y, float pixel, 
            float min_depth, float max_depth, depthHo *dh, cv::Mat F12, float &best_u, float &best_v, float th_pi)
    {

        // 极线方程参数。
        // 因为基本矩阵是F12，这里应该是F12的转置。
        float epipolar_a = F12.at<float>(0,0)*img_x + F12.at<float>(1,0)*img_y + F12.at<float>(2,0);
        float epipolar_b = F12.at<float>(0,1)*img_x + F12.at<float>(1,1)*img_y + F12.at<float>(2,1);
        float epipolar_c = F12.at<float>(0,2)*img_x + F12.at<float>(1,2)*img_y + F12.at<float>(2,2);

        // 如果极线接近与x垂直，舍弃。 
        if(abs(epipolar_a/epipolar_b) >4) 
            return ;

        
        float old_err = 1000.0;
        float best_photometric_err = 0.0;               // 最低灰度误差。
        float best_gradient_modulo_err = 0.0;           // 最低灰度梯度误差摸。
        int best_pixel = 0;                           // 最接近灰度。

        int vj, uj_plus, vj_plus, uj_minus, vj_minus;   // 用于计算论文中灰度导数q，梯度导数g。
        float g, q, denominator, ustar, ustar_var;

        // 获取极对匹配搜索范围。
        float umin(0.0), umax(0.0);
        GetSearchRange(umin, umax, img_x, img_y, min_depth, max_depth, pKF1, pKF2);

        // 遍历极线上的像素。
        for(int uj=floor(umin); uj<ceil(umax)+1; uj++)
        {
            // 极线方程求像素坐标vj。
            vj = -(int)((epipolar_a/epipolar_b)*uj+(epipolar_c/epipolar_b));
            if(vj<0 || vj>pKF2->im_.rows)
                continue;

            // 步骤1 极对搜索对应点判据。

            // condition 1: 像素梯度大于阈值。 
            if(pKF2->GradImg.at<float>(vj,uj) < lambdaG)
                continue;

            // condition2: 像素梯度方向不能与极线方向垂直。
            float epipolar_line_angle = cv::fastAtan2(-epipolar_a/ epipolar_b, 1);                 // 极线方向。
            float pixel_gradient_angle = pKF2->GradTheta.at<float>(vj, uj);                               // 像素梯度方向i。
            // float angle_abs = abs(pixel_gradient_angle - epipolar_line_angle);  // 夹角。

            // if( abs(90-angle_abs) <(90-lambdaL) || abs(angle_abs-180-90) < (90-lambdaL) )
            //     continue;

            if(pixel_gradient_angle > 270)
                pixel_gradient_angle = pixel_gradient_angle-360;

            if(pixel_gradient_angle>90 && pixel_gradient_angle <=270)
                pixel_gradient_angle = pixel_gradient_angle-180;

            if(epipolar_line_angle > 270)
                epipolar_line_angle = epipolar_line_angle-360;

            if(abs( abs(pixel_gradient_angle - epipolar_line_angle)-90 ) <10)
                continue;

             // condition3: 像素梯度方向应相似。
             if(abs(pKF2->GradTheta.at<float>(vj,uj) - th_pi) > lambdaTheta+10)             // 666
                 continue;

            // 6666
            float photometric_err = pixel - bilinear<uchar>(pKF2->im_, -((epipolar_a/epipolar_b)*uj+(epipolar_c/epipolar_b)), uj);
            float gradient_modulo_err = pKF1->GradImg.at<float>(img_y, img_x) - bilinear<float>(pKF2->GradImg, -((epipolar_a/epipolar_b)*uj+(epipolar_c/epipolar_b)), uj);
            float err = (photometric_err*photometric_err + (gradient_modulo_err*gradient_modulo_err)/THETA);
            // float err = (photometric_err*photometric_err + (gradient_modulo_err*gradient_modulo_err)/THETA)/(pKF2->I_stddev * pKF2->I_stddev);

            if(err < old_err)
            {
                best_pixel = uj;
                old_err = err;
                best_photometric_err = photometric_err;
                best_gradient_modulo_err = gradient_modulo_err;
            }
        }

        if(old_err < 500.0)
        {
            uj_plus = best_pixel+1;
            vj_plus = -((epipolar_a/epipolar_b)*uj_plus + (epipolar_c/epipolar_b));
            uj_minus = best_pixel-1;
            vj_minus = -((epipolar_a/epipolar_b)*uj_minus + (epipolar_c/epipolar_b));

            g = ((float)pKF2->im_.at<uchar>(vj_plus, uj_plus) - (float)pKF2->im_.at<uchar>(vj_minus, uj_minus))/2.0;
            q = (pKF2->GradImg.at<float>(vj_plus, uj_plus) - pKF2->GradImg.at<float>(vj_minus, uj_minus))/2.0;

            denominator = (g*g + (1.0/THETA)*q*q);
            ustar = best_pixel + (g*best_photometric_err + (1.0/THETA)*q*best_gradient_modulo_err)/denominator;     // 亚像素精度下的匹配像素坐标u。
            ustar_var = (pKF2->I_stddev * pKF2->I_stddev)/denominator;                                              // 匹配像素不确定性。
            
            best_u = ustar;
            best_v = -((epipolar_a/epipolar_b)*best_u +(epipolar_c/epipolar_b));

            ComputeInvDepthHypothesis(pKF1, pKF2, ustar, ustar_var, epipolar_a, epipolar_b, epipolar_c, dh, img_x, img_y);
        }

    }

    // 帧内深度检验。
    void ProbabilityMapping::IntraKeyFrameDepthChecking(cv::Mat &depth_map, cv::Mat &depth_sigma, const cv::Mat gradimg)
    {

        cv::Mat depth_map_new = depth_map.clone();
        cv::Mat depth_sigma_new = depth_sigma.clone();

        int grow_cnt(0);
	int fuse_cnt(0);
        for(int img_y=2; img_y<(depth_map.rows-2); img_y++)
        {
            for(int img_x=2; img_x<(depth_map.cols-2); img_x++)
            {
                // 梯度大的像素，增加重建地图点云密度。
                if(depth_map.at<float>(img_y, img_x) < 0.0001)
                {
                    // 跳过低梯度像素。
                    if(gradimg.at<float>(img_y, img_x)<lambdaG)
                        continue;

                    vector<pair<float, float> > max_supported;          // 深度一致的像素总数量

                    // 搜索当前像素附近的8个像素，最少有两个深度一致，增加重建效果。
                    
                    // 选择其中的一个像素。
                    for(int y1=img_y-1; y1<=img_y+1; y1++)
                    {
                        for(int x1=img_x-1; x1<=img_x+1; x1++)
                        {
                            vector<pair<float, float> > supported;
                            if(x1==img_x && y1==img_y)
                                continue;

                            // 遍历剩余元素，进行一致性检验。
                            for(int x2=img_x-1; x2<=img_x+1; x2++ )
                            {
                                for(int y2=img_y-1; y2<=img_y+1; y2++)
                                {
                                    if((x1==x2 && y1==y2) || (x2==img_x && y2==img_y))
                                        continue;

                                    if( ChiTest(depth_map.at<float>(y1, x1), depth_map.at<float>(y2, x2), depth_sigma.at<float>(y1,x1), depth_sigma.at<float>(y2,x2)) )
                                    {
                                        pair<float,float> depth;
                                        depth.first = depth_map.at<float>(y2, x2);
                                        depth.second = depth_sigma.at<float>(y2, x2);
                                        supported.push_back(depth);
					// supported.push_back(make_pair(depth_map.at<float>(y2,x2), depth_sigma.at<float>(y2,x2)) );
                                    }
                                }

                            }

                            // 保留有最多一致深度的像素。
                            if(supported.size()>0 && supported.size() > max_supported.size())
                            {
                                pair<float, float> depth;
                                depth.first = depth_map.at<float>(y1, x1);
                                depth.second = depth_sigma.at<float>(y1,x1);
                                supported.push_back(depth);
				// supported.push_back(make_pair(depth_map.at<float>(y1,x1), depth_sigma.at<float>(y1,x1)) );

                                max_supported = supported;
                            }
                            

                        }
                    }

                    if(max_supported.size() >1 )
                    {
                        grow_cnt ++;
                        float d(0.0), sigma(0.0);
                        GetFusion(max_supported, d, sigma);
                        depth_map_new.at<float>(img_y, img_x) = d;
                        depth_sigma_new.at<float>(img_y, img_x) = sigma;
                    }

                }

                else
                {
                    vector<pair<float, float> > supported;
                    supported.push_back(make_pair(depth_map.at<float>(img_y,img_x), depth_sigma.at<float>(img_y, img_x)) );

                    for(int y=img_y-1; y<=img_y+1; y++)
                    {
                        for(int x=img_x-1; x<=img_x+1; x++)
                        {

                            if(x==img_x && y==img_y)
                                continue;

                            if(depth_map.at<float>(y,x) > 0)
                            {
                                if(ChiTest(depth_map.at<float>(y,x), depth_map.at<float>(img_y, img_x), depth_sigma.at<float>(y,x), depth_sigma.at<float>(img_y, img_x)) )
                                {
                                    supported.push_back(make_pair(depth_map.at<float>(y,x), depth_sigma.at<float>(y,x)) );
                                }
                            }


                        }
                    }

                    if(supported.size()>3)
                    {
			fuse_cnt++;
                        float d(0.0), sigma(0.0);
                        GetFusion(supported, d, sigma);
                        depth_map_new.at<float>(img_y, img_x) = d;
                        depth_sigma_new.at<float>(img_y, img_x) = sigma;
                    }

                    else
                    {
                        depth_map_new.at<float>(img_y, img_x) = 0.0;
                        depth_sigma_new.at<float>(img_y, img_x) = 0.0;
                    }
                }


            }
        }

        // cout<<"intra key frame grow pixel number: "<<grow_cnt<<endl;
	// cout<<"intra key frame fuse pixel number: "<<fuse_cnt<<endl;

        depth_map = depth_map_new.clone();
        depth_sigma = depth_sigma_new.clone();

    }
    
   
  
    
    // 帧间逆深度假设检验。
    void ProbabilityMapping::InterKeyFrameDepthChecking(KeyFrame * currentKF)
    {
        vector<KeyFrame *> neighbors;

        neighbors = currentKF->GetBestCovisibilityKeyFrames(covisN);
        if(neighbors.size() < covisN)
            return;

        // 对于关键帧ki中的像素，投影到临近关键帧kj中计算逆深度。
        vector<cv::Mat> Rji, tji;
        for(size_t j=0; j<neighbors.size(); j++)
        {
            KeyFrame *pKF = neighbors[j];

            cv::Mat Rcw1 = currentKF->GetRotation();
            cv::Mat tcw1 = currentKF->GetTranslation();
            
            cv::Mat Rcw2 = pKF->GetRotation();
            cv::Mat tcw2 = pKF->GetTranslation();

            // Tcw2 = T21*Tcw1.
            cv::Mat R21 = Rcw2*Rcw1.t();
            cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

            Rji.push_back(R21);
            tji.push_back(t21);
        }

        int cols = currentKF->im_.cols;
        int rows = currentKF->im_.rows;
        float fx = currentKF->fx;
        float fy = currentKF->fy;
        float cx = currentKF->cx;
        float cy = currentKF->cy;
        int remove_cnt (0);

        for(int img_y=2; img_y<rows-2; img_y++)
        {
            for(int img_x=2; img_x<cols-2; img_x++)
            {

                // 深度==0
                if(currentKF->depth_map_.at<float>(img_y, img_x) < 0.0001)
                    continue;

                // 关键帧ki的像素p的逆深度。
                float depth_p = currentKF->depth_map_.at<float>(img_y, img_x);

                // 临近关键帧kj中的像素逆深度假设一致个数。
                int compatible_neighbor_keyframes_count = 0;

                for(size_t j=0; j<neighbors.size(); j++)
                {
                    KeyFrame *pKFj = neighbors[j];
                    cv::Mat K = pKFj->mK.clone();

                    // 反向投影到关键帧ki相机坐标系下。
                    cv::Mat xp = (cv::Mat_<float>(3,1) << (img_x-cx)/fx, (img_y-cy)/fy, 1.0);
                    cv::Mat temp = Rji[j]*xp/depth_p + tji[j];
                    cv::Mat Xj = K*temp;
                    float depth_j = 1.0/Xj.at<float>(2);
                    Xj = Xj/Xj.at<float>(2);    // 像素坐标。

                    // 文档方程12计算Xj的深度。
                    // float depth_j = depth_p/(Rji[j].row(2)*xp + depth_p*tji[j].at<float>(2));

                    float xj = Xj.at<float>(0);
                    float yj = Xj.at<float>(1);

                    // 搜索Xj附近的4个像素是否有一致的。
                    if(xj<0 || xj>cols || yj<0 || yj>rows)
                        continue;

                    int x0 = floor(xj);
                    int y0 = floor(yj);
                    vector<float> compatible_pixels;

                    // 搜索附近4个像素一致性。
                    for(int y1=y0; y1<=y0+1; y1++)
                    {
                        for(int x1=x0; x1<=x0+1; x1++)
                        {

                            float depth = pKFj->depth_map_.at<float>(y1,x1);
                            float sigma = pKFj->depth_sigma_.at<float>(y1,x1);

                            if(depth > 0.000001)
                            {
                                float test = pow((depth_j- depth), 2) / pow(sigma,2);

                                if(test < 3.84)
                                    compatible_pixels.push_back(depth);
                            }

                        }
                    }

                    if(compatible_pixels.size())
                        compatible_neighbor_keyframes_count++;
                }

                if(compatible_neighbor_keyframes_count < 1)
                {
                    currentKF->depth_map_.at<float>(img_y, img_x) = 0.0;
                    remove_cnt++;
                }

            }
        }

        cout<<"Inter Key Frame checking , remove outlier: "<<remove_cnt<< endl;

    }


    
    // 逆深度假设融合
    void ProbabilityMapping::InverseDepthHypothesisFusion(const vector<depthHo> &hypothes, depthHo &dist)
    {

        dist.depth = 0.0;
        dist.sigma = 0.0;
        dist.supported = false;

        float cnt = 0;
        vector<pair<float, float> > compatible_supported;
        vector<pair<float, float> > supported;

        for(size_t i=0; i<hypothes.size(); i++)
        {
	    
	    supported.clear();
            for(size_t j=0; j<hypothes.size(); j++)
            {


                // if(i==j)
                //    continue;

                if(ChiTest(hypothes[i].depth, hypothes[j].depth, hypothes[i].sigma, hypothes[j].sigma) )
                    supported.push_back(make_pair(hypothes[j].depth, hypothes[j].sigma) );

            }

            if(supported.size() >= lambdaN)
                compatible_supported.push_back(make_pair(hypothes[i].depth, hypothes[i].sigma) );
        }

        if(compatible_supported.size() >= lambdaN)
        {
            cnt++;
            dist.supported = true;
            GetFusion(compatible_supported, dist.depth, dist.sigma);
        }
        // cout << "hypothes fuse pixel number: "<< cnt << endl;
    }



    // 计算深度假设
    /**
    *   输入参数
    *       pKF1, pKF2          关键帧。
    *       ustar               亚像素精度下的匹配像素坐标
    *       ustar_var           匹配像素的逆深度不确定性。
    *       epipolar_*          极线方程系数。
    *       dh                  得到的你深度假设。
    *       img_x, img_y        像素坐标。
    **/
    void ProbabilityMapping::ComputeInvDepthHypothesis(KeyFrame *pKF1, KeyFrame *pKF2, float ustar, float ustar_var,
                                                        float epipolar_a, float epipolar_b, float epipolar_c, ProbabilityMapping::depthHo *dh, int img_x, int img_y)
    {

        // 匹配像素逆深度。
        float inv_pixel_depth =0.0;         
        // 利用文档的公式8计算像素逆深度。
        GetPixelDepth(ustar, img_x, img_y, pKF1, pKF2, inv_pixel_depth);


        // 文档公式9中计算逆深度不确定性的像素坐标。 
        float ustar_min = ustar - sqrt(ustar_var);
        float inv_depth_min = 0.0;
        GetPixelDepth(ustar_min, img_x, img_y, pKF1, pKF2, inv_depth_min);

        float ustar_max = ustar + sqrt(ustar_var);
        float inv_depth_max = 0.0;
        GetPixelDepth(ustar_max, img_x, img_y, pKF1, pKF2, inv_depth_max);

        float sigma_depth = cv::max(abs(inv_depth_max-inv_pixel_depth), abs(inv_depth_min-inv_pixel_depth));

        dh->depth = inv_pixel_depth;
        dh->sigma = sigma_depth;
        dh->supported = true;

    }




    // 计算像素逆深度。 
    // 文档公式8
    void ProbabilityMapping::GetPixelDepth(float uj, int img_x, int img_y, KeyFrame *pKF1, KeyFrame *pKF2, float &inv_depth)
    {

        float fx = pKF1->fx;
        float cx = pKF1->cx;
		float fy = pKF1->fy;
        float cy = pKF1->cy;

        cv::Mat Rcw1 = pKF1->GetRotation();
        cv::Mat tcw1 = pKF1->GetTranslation();
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat tcw2 = pKF2->GetTranslation();

        cv::Mat R21 = Rcw2*Rcw1.t();
        cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;
        cv::Mat xp = (cv::Mat_<float>(3,1) << (img_x-cx)/fx, (img_y-cy)/fy, 1.0);       // 反向投影到pKF1相机坐标系下。
        
        // 计算像素逆深度，三角化。
        float ucx = uj-cx;
        cv::Mat temp = R21.row(2)*xp*ucx - fx*(R21.row(0)*xp);
        float numerator = temp.at<float>(0,0);
        float denominator = -t21.at<float>(2)*ucx + fx*t21.at<float>(0);

        inv_depth = numerator / denominator; 
    }



    // 计算极线匹配的搜索范围。
    void ProbabilityMapping::GetSearchRange(float &umin, float &umax, int img_x, int img_y, float min_depth, float max_depth,
											 KeyFrame *pKF1, KeyFrame *pKF2)
    {

        float fx = pKF1->fx;
        float fy = pKF1->fy;
        float cx = pKF1->cx;
        float cy = pKF1->cy;

        cv::Mat Rcw1 = pKF1->GetRotation();
        cv::Mat tcw1 = pKF1->GetTranslation();
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat tcw2 = pKF2->GetTranslation();

        cv::Mat R21 = Rcw2*Rcw1.t();
        cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;
        cv::Mat xp1 = (cv::Mat_<float>(3,1) << (img_x-cx)/fx, (img_y-cy)/fy, 1.0);       // 反向投影到pKF1相机坐标系下，归一化不带深度。

        if(min_depth < 0 )
            min_depth = 0;

        cv::Mat xp2_min = R21*(xp1*min_depth)+t21;
        cv::Mat xp2_max = R21*(xp1*max_depth)+t21;

        umin = fx*xp2_min.at<float>(0)/xp2_min.at<float>(2) + cx;
        umax = fx*xp2_max.at<float>(0)/xp2_max.at<float>(2) + cx;

        if(umin < 0)
            umin = 0;

        if(umax < 0)
            umax = 0;
        
        if(umin > pKF1->im_.cols) 
            umin = pKF1->im_.cols;
        
        if(umax > pKF1->im_.cols)
            umax = pKF1->im_.cols;

    }



    bool ProbabilityMapping::ChiTest(const float &depth1, const float &depth2, const float sigma1, const float sigma2)
    {
        float numerator = (depth1-depth2)*(depth1-depth2);
        float chi_test = numerator / ( sigma1*sigma1) + numerator/(sigma2*sigma2);

        return (chi_test < 5.991);      // 95%的卡方分布。
    }



    void ProbabilityMapping::GetFusion(const vector<pair<float, float> > supported, float &depth, float &sigma)
    {

        float sum_depth = 0;
        float sum_sigma = 0;

        for(size_t i=0; i< supported.size(); i++)
        {

            sum_depth += supported[i].first / pow(supported[i].second, 2);
            sum_sigma += 1/pow(supported[i].second, 2);
        }

        depth = sum_depth / sum_sigma;
        sigma = sqrt(1.0/ sum_sigma);

    }





    // 计算两帧之间的基本矩阵。
    cv::Mat ProbabilityMapping::ComputeFundamental(KeyFrame *pKF1, KeyFrame *pKF2)
    {

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w*R2w.t();
        cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
        cv::Mat t12x = GetSkewSymmetricMatrix(t12);

        cv::Mat K1 = pKF1->GetCalibrationMatrix();
        cv::Mat K2 = pKF2->GetCalibrationMatrix();

        return K1.t().inv()*t12x*R12*K2.inv();

    }
    

    cv::Mat ProbabilityMapping::GetSkewSymmetricMatrix(const cv::Mat &vec)
    {
        return (cv::Mat_<float>(3,3) <<              0, -vec.at<float>(2), vec.at<float>(1),
                                        vec.at<float>(2),               0, -vec.at<float>(0),
                                        -vec.at<float>(1), vec.at<float>(0),               0);
    }



}




#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>


namespace ORB_SLAM2
{

    // 构造函数，初始化。
    FrameDrawer::FrameDrawer(Map *pMap):mpMap(pMap)
    {
        mState = Tracking::SYSTEM_NOT_READY;

        // 存储用于画图的Frame信息。
        // 包括：图像 特征点连线形成的轨迹（初始化时）框（跟踪时的MapPoint）圈（跟踪时的特点）。
        mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    }



    // 准备需要显示的信息，图像，状态，其他提示。
    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys;          // 初始化：参考帧的特征点。
        vector<int> vMatches;                    // 初始化：参考特征点的对应匹配。
        vector<cv::KeyPoint> vCurrentKeys;       // 当前帧的特征点。       
        vector<bool> vbVO, vbMap;               // 当前帧的跟踪特征点。
        int state;                              // 跟踪状态。

        // 成员变量复制给局部变量，加互斥锁。
        {
            unique_lock<mutex> lock(mMutex);
            state = mState;
            if(mState == Tracking::SYSTEM_NOT_READY)
                mState = Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            if(mState == Tracking::NOT_INITIALIZED)
            {
                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
            }
            else if(mState == Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
            }
            else if(mState == Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
            }

        }

		if(im.channels() <3 )
			cvtColor(im, im, CV_GRAY2BGR);
        // 初始化时，当前帧窗口的特征坐标与初始帧的特征点坐标连成线，形成初始化时的轨迹线。
        if(state == Tracking::NOT_INITIALIZED)
        {
            for(unsigned int i=0; i<vMatches.size(); i++)
            {
                if(vMatches[i] >= 0)
                {
                    cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt, cv::Scalar(0,0,255));
                }
            }
        }

        // 跟踪时，
        else if(state == Tracking::OK)
        {
            // 跟踪特征点数量。
            mnTracked = 0;
            mnTrackedVO = 0;

            // 绘制特征点。
            const float r=5;
            for(int i=0; i<N; i++)
            {
                if(vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1, pt2;
                    pt1.x = vCurrentKeys[i].pt.x - r;   
                    pt1.y = vCurrentKeys[i].pt.y - r;
                    pt2.x = vCurrentKeys[i].pt.x + r;
                    pt2.y = vCurrentKeys[i].pt.y + r;

                    // SLAM模式下，地图中匹配的地图点云。
                    if(vbMap[i])
                    {
                        // 红色框+点，表示当前帧中MapPoints对应的特征点。
                        cv::rectangle(im, pt1, pt2, cv::Scalar(0,0,255));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0,0,255), -1);
                        mnTracked++;
                    }
                    // 定位模式下，地图中匹配的点云。
                    else
                    {
                        cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0,255,0), -1);
                        mnTrackedVO++;
                    }
                }
            }
        }

        cv::Mat imWithInfo;
        DrawTextInfo(im, state, imWithInfo);

        return imWithInfo;

    }


    // 当前帧窗口下面的文字。
    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        if(nState == Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if(nState == Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if(nState == Tracking::OK)
        {
            if(!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";

            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
            if(mnTrackedVO > 0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if(nState == Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if(nState == Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
        
        imText = cv::Mat(im.rows+textSize.height+10, im.cols, im.type());
        im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
        imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height+10, im.cols, im.type());
        cv::putText(imText, s.str(), cv::Point(5, imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255),1,8);

    }



    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N, false);
        mvbMap = vector<bool>(N,false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        if(pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED)
        {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        }
        else if(pTracker->mLastProcessedState == Tracking::OK)
        {
            for(int i=0; i<N; i++)
            {
                MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if(!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        if(pMP->Observations() >0)
                            mvbMap[i] = true;
                        else 
                            mvbVO[i] = true;
                    }
                }
            }
        }
		
		mState = static_cast<int>(pTracker->mLastProcessedState);
    }





}   // namespace ORB_SLAM2






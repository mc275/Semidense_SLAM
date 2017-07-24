 



#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ProbabilityMapping.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{

    MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath): mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        // 读取地图中点云，关键帧，相机，图连线的显示大小。
        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

    }



    // 绘制地图中的点云。
    void MapDrawer::DrawMapPoints()
    {
        const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
        const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

        set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if(vpMPs.empty())
            return;

        // 对于所有的地图点云，显示为黑色。
        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0, 0.0, 0.0);

        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            // 不包括坏点和参考点。
            if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                continue;

            cv::Mat pos = vpMPs[i]->GetWorldPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2) );
        }
	
	glEnd();
        
	// 参考点显示为红色。
        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(1.0, 0.0, 0.0);

        for(set<MapPoint *>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;

            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }
        glEnd();
 }

void MapDrawer::DrawSemiDense()
{

    const vector<KeyFrame*> &vpKF = mpMap->GetAllKeyFrames();
    if(vpKF.empty())
      return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,1.0,0.0);

    
    int draw_cnt(0);
    for(size_t i = 0; i < vpKF.size();++i)
    {
        KeyFrame* pKF = vpKF[i];
        
	if(! pKF->semidense_flag_) 
	  continue;
        
	draw_cnt ++;
	
        for(size_t y = 0; y< (size_t)(pKF->im_.rows); y++)
          for(size_t x = 0; x< (size_t)(pKF->im_.cols); x++)
        {

          Eigen::Vector3f Pw  (pKF->SemiDensePointSets_.at<float>(y,3*x), pKF->SemiDensePointSets_.at<float>(y,3*x+1), pKF->SemiDensePointSets_.at<float>(y,3*x+2));
	  
          if(Pw[2]>0)
          {
            float b = pKF->rgb_.at<uchar>(y,3*x) / 255.0;
            float g = pKF->rgb_.at<uchar>(y,3*x+1) / 255.0;
            float r = pKF->rgb_.at<uchar>(y,3*x+2) / 255.0;
            glColor3f(r,g,b);
            glVertex3f( Pw[0],Pw[1],Pw[2]);
          }
        }
    }
    
    glEnd();

}
    



    // 绘制地图中的关键帧。
    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
    {
        const float &w = mKeyFrameSize;
        const float h = w*0.75;
        const float z = w*0.6;

        const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

        if(bDrawKF)
        {
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame *pKF = vpKFs[i];

                // 获取从相机坐标系到世界坐标系的位姿，并求转置(OpenGL中先存列)。
                cv::Mat Twc = pKF->GetPoseInverse().t();

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f, 0.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);
                
                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }
        }

        if(bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
            glBegin(GL_LINES);

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                // Covisibility图。 
                const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                if(!vCovKFs.empty())
                {
                    for(vector<KeyFrame *>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                    {
                        if( (*vit)->mnId < vpKFs[i]->mnId)
                            continue;
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
                    }
                }

                // 生成树 spanning tree。
                KeyFrame *pParent = vpKFs[i]->GetParent();
                if(pParent)
                {
                    cv::Mat Owp = pParent->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
                }

                // 闭环。
                set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
                for(set<KeyFrame *>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
                {
                    if((*sit)->mnId < vpKFs[i]->mnId)
                        continue;
                    cv::Mat Owl = (*sit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
                }
            }

            glEnd();
        }
    }



    // 绘制当前相机的位置。
    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {

        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

    #ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
    #else
        glMultMatrixd(Twc.m);
    #endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);

        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);

        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();

        glPopMatrix();
    }



    // 设置当前的相机位姿。
    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }



    // 当前相机到世界坐标系的变换矩阵。 
    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        if(!mCameraPose.empty())
        {
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<float>(0,0);
            M.m[1] = Rwc.at<float>(1,0);
            M.m[2] = Rwc.at<float>(2,0);
            M.m[3] = 0.0;

            M.m[4] = Rwc.at<float>(0,1);
            M.m[5] = Rwc.at<float>(1,1);
            M.m[6] = Rwc.at<float>(2,1);
            M.m[7] = 0.0;

            M.m[8] = Rwc.at<float>(0,2);
            M.m[9] = Rwc.at<float>(1,2);
            M.m[10] = Rwc.at<float>(2,2);
            M.m[11] = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15] = 1.0;
        }

        else
            M.SetIdentity();

    }




}   // namespace ORB_SLAM2








// 定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{
    class LoopClosing;

    class Optimizer
    {
        public:
            // 无构造函数。

            // 3D-2D最小化重投影误差。
            void static BundleAdjustment(const std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP, int nIterations=5,
                                            bool *pbStopFlag=NULL, const unsigned long nLoopKF=0, const bool bRobust=true );

            // 
            void static GlobalBundleAdjustment(Map *pMap, int nIterations=5, bool *pbStopFlag=NULL, 
                                                const unsigned long nLoopKF=0, const bool bRobust=true);

            //
            void static LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap);

            //3D-2D最小化重投影误差，只优化位姿。 
            int static PoseOptimization(Frame *pFrame);

            
            // 闭环检测后，EssentialGraph优化。
            // 如果bFixScale==true, 对于双目和RGBD来说运行6DoF优化，对于单目运行7DoF优化。
            void static OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF, 
                                                const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                                const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                                const map<KeyFrame *, set<KeyFrame *>> &LoopConnections,
                                                const bool &bFixScale);


            // 形成闭环时，Sim3优化。
            // 如果bFixScale==true, 对于双目和RGBD来说运行SE3优化，对于单目运行Sim3优化。
            static int OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches1, 
                                        g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);

    };

}   // namespace ORB_SLAM2





#endif

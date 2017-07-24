//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include<mutex>

namespace ORB_SLAM2
{
    
    class KeyFrame;
    class Frame;

    class KeyFrameDatabase
    {
        public:
            // 构造函数。
            KeyFrameDatabase(const ORBVocabulary &voc);
            
            // 添加关键帧。
            void add(KeyFrame *pKF);
            // 剔除关键帧。
            void erase(KeyFrame *pKF);
            
            void clear();

            // 闭环检测。
            std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame *pKF, float minScore);
            // 重定位。
            std::vector<KeyFrame *> DetectRelocalizationCandidates(Frame *F);

        protected:

            // 关联词典。
            // 预先训练好的词典。
            const ORBVocabulary *mpVoc;

            // 倒排文件。
            // 倒排索引，mvInvertedFile[i]表示包含第i个word id的所有关键帧。
            std::vector< list<KeyFrame *> > mvInvertedFile;

            // mutex
            std::mutex mMutex;
   
    };

}   // namespace ORB_SLAM2


#endif


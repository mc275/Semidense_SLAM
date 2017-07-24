


#include "KeyFrameDatabase.h"
#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include <mutex>


using namespace std;

namespace ORB_SLAM2
{

    // 构造函数。
    KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary &voc): mpVoc(&voc)
    {
        // 词的数量。
        mvInvertedFile.resize(voc.size());
    }



    // 根据关键帧的词包，更新数据库的倒排索引。
    void KeyFrameDatabase::add(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutex);

        // 为该KeyFrame包含的词添加关联， vit->first是关键帧pFK包含的词。
        for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
            mvInvertedFile[vit->first].push_back(pKF);
    }



    // 关键帧被删除后，更新数据库的倒排索引。
    void KeyFrameDatabase::erase(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutex);

        // 每一个pKF包含多个word，遍历mvInvertedFile中的words，根据word删除对应的pKF。
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        {
            // 列出包含又共同word的关键帧。 
            list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

            for(list<KeyFrame *>::iterator lit=lKFs.begin(), lend=lKFs.end(); lit!=lend; lit++)
            {
                if(pKF == *lit)
                {
                    lKFs.erase(lit);
                    break;
                }
            }
        }

    }



    // 清除KeyFrameDB。
    void KeyFrameDatabase::clear()
    {
        // mvInvertedFile[i] 表示包含第i个word id的所有关键帧。
        mvInvertedFile.clear(); 
        // 预先训练好的词典，就那个半天也加载不完的玩意。
        mvInvertedFile.resize(mpVoc->size());
    }



    /*
    * 在闭环检测中找到与该关键帧可能闭环的关键帧。
    *   1. 找出和当前关键帧具有公共word最多的关键帧。
    *   2. 只和具有共同单词较多的关键帧进行相似度计算。
    *   3. 将与共视较多关键帧相连(权值最高)的前十个关键帧归为一组，计算累计得分。
    *   4. 只返回累计的得分较高的组中分数最高的关键帧。
    * @param
    *   pKF 当前关键真，需要闭环的关键帧。
    *   minScore 相似分数最低要求。
    * @return    可能闭环的关键帧，一般不是一个。
    */
    vector<KeyFrame *> KeyFrameDatabase::DetectLoopCandidates(KeyFrame *pKF, float minScore)
    {
        // 获得与pKF相连的KF，这些KeyFram都是局部相连，闭环检测时需要剔除。
        set<KeyFrame *> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
        // 保存可能与pKF形成回环的关键帧(有相同的word，不是局部连接)。
        list<KeyFrame *> lKFsSharingWords;

        // 步骤1 找到和pKF具有公共word的所有关键帧，不包含spConnectedKeyFrames。
        {
            unique_lock<mutex> lock(mMutex);

            // words是检测图像是否匹配的关键，遍历pKF的每一word。
            for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
            {
                // 提取包含该word的所有KeyFrame。
                list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                // 遍历有相同word的KF。
                for(list<KeyFrame *>::iterator lit=lKFs.begin(), lend=lKFs.end(); lit!=lend; lit++)
                {
                    KeyFrame *pKFi=*lit;
                    // pKFi没有被标记为pKF的闭环候选帧。
                    if(pKFi->mnLoopQuery != pKF->mnId)
                    {
                        pKFi->mnLoopWords=0;
                        // pKFi不在局部相连中。
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            // 标记为pKF的候选。
                            pKFi->mnLoopQuery = pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }

                    pKFi->mnLoopWords++;
                }
            }
        }

        if(lKFsSharingWords.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *>> lScoreAndMatch;


        // 步骤2 统计所有闭环候选帧中与pKF具有最多word的关键帧对应的word数目。
        int maxCommonWords = 0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
        {
            if((*lit)->mnLoopWords > maxCommonWords)
                maxCommonWords = (*lit)->mnLoopWords;
        }

        // 计算相似度的word数目阈值。
        int minCommonWords = maxCommonWords*0.8f;
        
        int nscores = 0;

        // 步骤3 遍历所有候选关键帧，找出共有word数目大于minCommonwords且单词匹配度大于minScore的关键帧，存入lScoreAndMatch。
        for(list<KeyFrame *>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
        {
            KeyFrame *pKFi = *lit;

            // pKF只和共有words数目大于minCommonWords的关键帧进行比较。
            if(pKFi->mnLoopWords > minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
                pKFi->mLoopScore = si;

                // 大于最小匹配度minScore。
                if(si >= minScore)
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
            }
        }

        if(lScoreAndMatch.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *>> lAccScoreAndMatch;
        float bestAccScore = minScore;

        // 步骤4 lScoreAndMatch中每一个KF把自己共视程度较高的前10个关键帧化为一组；
        //       每一组会计算全组得分，并记录该组得分最高的KF，存在lAccScoreAndMatch中。
        for(list<pair<float, KeyFrame *>>::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
        {
            KeyFrame *pKFi = it->second;
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;    // 该组最高得分。
            float accScore = it->first;     // 该组累计得分。
            KeyFrame *pBestKF = pKFi;       // 该组最高得分对应KF。
            for(vector<KeyFrame *>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
            {
                // 候选帧的相邻帧。
                KeyFrame *pKF2 = *vit;
                // 只有pKF2在闭环候选帧lScoreAndMatch中，才能贡献分数。
                if(pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords)
                {
                    accScore += pKF2->mLoopScore;
                    // 组内最高得分和对应关键帧。
                    if(pKF2->mLoopScore>bestScore)
                    {
                        pBestKF = pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            // 所有组中的最高得分。
            if(accScore>bestAccScore)
                bestAccScore = accScore;
        }

        float minScoreToRetain = 0.75f*bestAccScore;

        // 已添加到候选vpLoopCandidates中的KF组成的数组，防止重复添加。
        set<KeyFrame *> spAlreadyAddedKF;
        // 闭环候选关键帧。
        vector<KeyFrame *> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        // 步骤5 得到组得分>minScoreToRetain的组，得到组中分数最高的关键帧。
        for(list<pair<float, KeyFrame *>>::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
        {
            if(it->first > minScoreToRetain)
            {
                KeyFrame *pKFi = it->second;
                // 防止重复添加候选帧。
                if(!spAlreadyAddedKF.count(pKFi))
                {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpLoopCandidates;

    }



    /**
    * 在重定位中找到与该帧相似的关键帧
    *   1.找出和当前帧具有公共单词的所有关键帧
    *   2.只和具有共同单词较多的关键帧进行相似度计算
    *   3.将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
    *   4.只返回累计得分较高的组中分数最高的关键帧
    * @param F 需要重定位的帧
    * @return  相似的关键帧。
    */
    vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
    {
        // 相比于关键帧闭环监测 DetectLoopCandidates()，帧F不存在共视图。
        // 可能和帧F形成重定位的候选关键帧。
        list<KeyFrame *> lKFsSharingWords;

        // 步骤1 找出和当前帧具有公共words的所有关键帧。
        {
            unique_lock<mutex> lock(mMutex);

            for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit!=vend; vit++)
            {
                // 提取包含该word的所有KeyFrame。
                list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                // 遍历有相同word的KF。
                for(list<KeyFrame *>::iterator lit=lKFs.begin(), lend=lKFs.end(); lit!=lend; lit++)
                {
                    KeyFrame *pKFi=*lit;
                    // pKFi没有被标记为F的重定位候选帧。
                    if(pKFi->mnRelocQuery != F->mnId)
                    {
                        pKFi->mnRelocWords=0;
                        pKFi->mnRelocQuery = F->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }

                    pKFi->mnRelocWords++;
                }
            }
        }

        if(lKFsSharingWords.empty())
            return vector<KeyFrame *>();

        // 步骤2 统计所有重定位候选帧中与当前帧F具有最多word的关键帧对应的word数目。
        int maxCommonWords = 0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
        {
            if((*lit)->mnRelocWords > maxCommonWords)
                maxCommonWords = (*lit)->mnRelocWords;
        }

        // 计算相似度的word数目阈值。
        int minCommonWords = maxCommonWords*0.8f;

        list<pair<float,KeyFrame*> > lScoreAndMatch;
        
        int nscores = 0;

        // 步骤3 遍历所有候选关键帧，找出共有word数目大于minCommonwords的关键帧，存入lScoreAndMatch。
        for(list<KeyFrame *>::iterator lit=lKFsSharingWords.begin(), lend=lKFsSharingWords.end(); lit!=lend; lit++)
        {
            KeyFrame *pKFi = *lit;

            // pKF只和共有words数目大于minCommonWords的关键帧进行比较。
            if(pKFi->mnRelocWords > minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
                pKFi->mRelocScore = si;
                lScoreAndMatch.push_back(make_pair(si,pKFi));
            }
        }

        if(lScoreAndMatch.empty())
            return vector<KeyFrame*>();

        list<pair<float,KeyFrame*> > lAccScoreAndMatch;
        float bestAccScore = 0;
        
        // 步骤4 lScoreAndMatch中每一个KF把自己共视程度较高的前10个关键帧化为一组；
        //       每一组会计算全组得分，并记录该组得分最高的KF，存在lAccScoreAndMatch中。
        for(list<pair<float, KeyFrame *>>::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
        {
            KeyFrame *pKFi = it->second;
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;    // 该组最高得分。
            float accScore = bestScore;     // 该组累计得分。
            KeyFrame *pBestKF = pKFi;       // 该组最高得分对应KF。
            for(vector<KeyFrame *>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
            {
                // 候选帧的相邻帧。
                KeyFrame *pKF2 = *vit;
                // 只有pKF2在闭环候选帧lScoreAndMatch中，才能贡献分数。
                if(pKF2->mnRelocQuery != F->mnId)
                    continue;

                accScore += pKF2->mRelocScore;
                // 组内最高得分和对应关键帧。
                if(pKF2->mRelocScore > bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mRelocScore;
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            // 所有组中的最高得分。
            if(accScore>bestAccScore)
                bestAccScore = accScore;
        }


        float minScoreToRetain = 0.75f*bestAccScore;

        // 已添加到候选vpLoopCandidates中的KF组成的数组，防止重复添加。
        set<KeyFrame *> spAlreadyAddedKF;
        // 闭环候选关键帧。
        vector<KeyFrame *> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());

        // 步骤5 得到组得分>minScoreToRetain的组，得到组中分数最高的关键帧。
        for(list<pair<float, KeyFrame *>>::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
        {
            const float &si = it->first;

            if(si > minScoreToRetain)
            {
                KeyFrame *pKFi = it->second;
                // 防止重复添加候选帧。
                if(!spAlreadyAddedKF.count(pKFi))
                {
                    vpRelocCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpRelocCandidates;

    }



}   // namespace ORB_SLAM2




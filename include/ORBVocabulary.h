//定义预处理变量，#ifndef variance_name 表示变量未定义时为真，并执行之后的代码直到遇到 #endif。
#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H


#include"Thirdparty/DBoW2/DBoW2/FORB.h"
#include"Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace ORB_SLAM2
{
    // 对DBoW2中的类重命名
    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

}   //namespace ORB_SLAM2


#endif  //ORBVOCABULARY_H


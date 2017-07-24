// Tum数据集测试ORB_SLAM。

// "" 表示程序目录相对路径，<>表示编译器库目录路径。
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>

#include <System.h>

// 声明标准库命名空间。
using namespace std;

// 加载Tum数据集。
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &Timestamps);

// 程序入口。
int main (int argc, char **argv)
{   
    // 判断输入参数个数是否满足格式。
    if(argc!=4)
    {
        cerr<< endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" <<endl;
        return 1;
    }

    // 提取数据集图像路径。
    vector<string> vstrImageFilenames;      // 帧图像名称
    vector<double> vTimestamps;             // 数据集时间戳。
    // argv[3]是数据集rgb.tx文件所在目录。
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    // 数据集帧数。
    int nImages = vstrImageFilenames.size();

    // 创建SLAM系统，初始化所有系统线程准备处理帧图像。
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // 完成一帧图像的静态处理时间。
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "......" << endl;
    cout << "Start processing sequence ... " << endl;
    cout << "Images in the sequence: " << nImages << endl;


    
    cv::Mat im;
	// 主循环。
    for(int ni=0; ni<nImages; ni++)
    {
        // 从文件中读取的帧图像。
        im = cv::imread( string(argv[3])+"/"+vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED );
        // 读取的当前图像对应的时间戳。
        double tframe = vTimestamps[ni];

        // 如果没有读取到图像。
        if( im.empty( ) )
        {
            cerr << endl << "Failed to load image at :"
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }


#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif


        // SLAM对图像进行处理。
        SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif


        // 处理单帧图像时间。
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2-t1).count();
        // 保存处理时间。
        vTimesTrack[ni]=ttrack;



        // 等待加载下一帧图像的时间，
        double T=0;
        // 没有读取到最后一帧图像。
        if (ni< nImages-1)
           // 计算采集时的时间间隔。 
            T=vTimestamps[ni+1]-tframe;
        // 读取到最后一帧图像。
        else if (ni>0)
           // 使用上一次的间隔时间。
            T=tframe-vTimestamps[ni-1];

        // 处理数据快于图像采集，延时等待加载。
        if (ttrack < T)
            usleep((T-ttrack)*1e6);
      
    }

    // 回车后继续执行。
    cout << "Press Enter to Shutdown the System." << endl; 
    getchar();  


    // 终止所有线程。
    SLAM.Shutdown();

    // 跟踪时间统计。
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime=0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }

    cout << "........" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean Tracking time: " << totaltime/nImages << endl;
    
    // 保存相机轨迹。
    SLAM.SaveTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;

}

/**********
*   strFile 表示Tum数据集中rgb.txt文件的完整路径。
*   vstrImageFilenames 表示rgb.txt中rgb/帧图像名字。
*   vTimestamps 表示rgb.txt中的时间戳。
  *********/

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{

    ifstream f;
    f.open(strFile.c_str());

    
    string s0;
    // 跳过rgb.txt前三行的说明。
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    // 获取文件名和时间戳。
    while(!f.eof())         // eof是否到文件尾部。
    {
        string s;
	// 读取f文件中的一行。
        getline(f,s);

        if(!s.empty())
        {
            stringstream ss;
            ss << s;			// 向一个string流写入输出数据，全部写入。
            double t;
            string sRGB;
            ss >> t;			// 从string流读取输入数据，遇到空白符停止。
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }

    }

}



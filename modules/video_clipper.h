#ifndef VIDEO_CLIPPER_H
#define VIDEO_CLIPPER_H

#include "target_processing.h" // 包含存储目标人物出现的时间片段的结构体的头文件（重复定义使用）
#include <opencv2/opencv.hpp> // 包含 OpenCV 库的头文件
#include <vector>            // 包含动态数组容器的头文件
#include <string>            // 包含字符串类的头文件

// 视频剪辑类：负责根据给定的时间片段剪辑视频
class VideoClipper {
public:
    // 构造函数
    VideoClipper(const std::string& input_path,  // 输入视频文件路径
                 const std::string& output_dir);  // 输出视频片段的目录

    // 剪辑视频片段
    // 参数：
    //   clips: 包含目标人物出现时间片段的向量
    //   fps: 视频的帧率
    //   frame_size: 视频的帧尺寸
    //   codec: 视频编码格式
    // 返回值：
    //   bool: 如果剪辑成功，则返回 true，否则返回 false
    bool clipVideo(const std::vector<TargetClip>& clips, double fps, cv::Size frame_size, int codec);

private:
    std::string input_path_;  // 输入视频文件路径
    std::string output_dir_; // 输出视频片段的目录
};

#endif // VIDEO_CLIPPER_H
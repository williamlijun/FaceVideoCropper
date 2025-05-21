#ifndef VIDEO_IO_H
#define VIDEO_IO_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <filesystem>


// 视频输入输出类，负责读取视频帧和写入裁剪后的视频。
class VideoIO {
public:
    VideoIO(const std::string& input_path, const std::string& output_path);
    ~VideoIO();

    static bool createDirectory(const std::string& folderPath); // 创建指定的文件夹

    static bool copyFile(const std::string& sourceFilePath, const std::string& destinationFolder); // 将源路径的文件复制到目标文件夹中

    bool open(); // 打开视频

    bool isOpened() const; // 检查视频是否已打开

    bool readFrame(cv::Mat& frame); // 读取下一帧

    double getFPS() const; // 获取视频帧率

    cv::Size getFrameSize() const; // 获取视频size

    long getTotalFrames() const; // 获取视频总帧数

    int getFourcc() const; // 获取视频编码格式 (FourCC)

    double getDuration() const; // 获取视频总时长 (秒)

    std::string getFourccString() const; // 获取可读的视频编码格式字符串

    // bool setFramePosition(int frameNumber); // 设置视频的帧位置

    void release(); // 释放视频捕获资源

private:
    std::string input_path_;   // 输入视频文件路径
    std::string output_path_;  // 输出视频文件路径
    cv::VideoCapture cap_;     // 视频捕获对象，用于读取视频
};

#endif // VIDEO_IO_H
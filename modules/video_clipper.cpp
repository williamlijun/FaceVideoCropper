#include "video_clipper.h"
#include <opencv2/opencv.hpp> // 包含 OpenCV 库的头文件
#include <iostream>          // 包含输入输出流的头文件
#include <sstream>          // 包含字符串流的头文件
#include <fstream>          // 包含文件输入输出流的头文件
#include <cstdlib>          // 包含通用工具函数的头文件

// 构造函数
VideoClipper::VideoClipper(const std::string& input_path, const std::string& output_dir)
    : input_path_(input_path), output_dir_(output_dir) {}

// 剪辑视频片段
bool VideoClipper::clipVideo(const std::vector<TargetClip>& clips, double fps, cv::Size frame_size, int codec) {
    // 如果没有需要剪辑的片段，则直接返回
    if (clips.empty()) {
        std::cout << "\n未检测到目标人物，无需剪辑视频片段。" << std::endl;
        return true;
    }

    std::cout << "\n开始剪辑包含目标人物的视频片段..." << std::endl;
    cv::VideoCapture tracked_video(input_path_); // 打开原始视频
    if (!tracked_video.isOpened()) {
        std::cerr << "无法打开已保存的跟踪视频文件: " << input_path_ << std::endl;
        return false;
    }

    std::ofstream concat_list_file(output_dir_ + "/concat_list.txt"); // 创建用于存储 FFmpeg 合并列表的临时文件
    if (!concat_list_file.is_open()) {
        std::cerr << "无法创建临时文件 " << output_dir_ << "/concat_list.txt" << std::endl;
        tracked_video.release();
        return false;
    }

    // 遍历所有记录的目标人物出现的时间片段
    for (size_t i = 0; i < clips.size(); ++i) {
        int start_frame = clips[i].start_frame; // 获取片段的起始帧
        int end_frame = clips[i].end_frame;     // 获取片段的结束帧
        std::stringstream ss;
        ss << "clip_" << i << ".avi"; // 生成片段的文件名
        std::string clip_filename = output_dir_ + "/" + ss.str();             // 生成完整的片段保存路径
        std::string concat_list_file_filename = ss.str(); // 获取片段文件名，用于 FFmpeg 合并列表

        // 使用 OpenCV 保存每个片段 (从视频中获取)
        cv::VideoWriter clip_writer(clip_filename, codec, fps, frame_size, true);
        if (clip_writer.isOpened()) {
            tracked_video.set(cv::CAP_PROP_POS_FRAMES, start_frame); // 设置读取的起始帧
            cv::Mat clip_frame;
            // 逐帧读取并写入当前片段
            for (int j = start_frame; j <= end_frame; ++j) {
                if (tracked_video.read(clip_frame)) {
                    clip_writer.write(clip_frame); // 将目标人物帧写入片段文件
                } else {
                    std::cerr << "读取跟踪视频帧失败，无法写入剪辑片段。" << std::endl;
                    break;
                }
            }
            clip_writer.release(); // 释放片段写入器
            concat_list_file << "file '" << concat_list_file_filename << "'\n"; // 将片段文件名写入合并列表文件
            std::cout << "已保存片段: " << clip_filename << " (帧 " << start_frame << " - " << end_frame << ")" << std::endl;
        } else {
            std::cerr << "无法打开文件以保存剪辑片段: " << clip_filename << std::endl;
        }
    }

    concat_list_file.close();   // 关闭合并列表文件
    tracked_video.release();    // 释放视频的读取器

    // 使用 FFmpeg 合并片段
    std::string output_clip_path = output_dir_ + "/target_clips.avi"; // 定义合并后的视频文件名
    std::string ffmpeg_command = "ffmpeg -f concat -safe 0 -i " + output_dir_ + "/concat_list.txt -c copy \"" + output_clip_path + "\"";
    int ffmpeg_result = system(ffmpeg_command.c_str()); // 执行 FFmpeg 命令
    if (ffmpeg_result == 0) {
        std::cout << "\n已成功剪辑包含目标人物的视频并保存到: " << output_clip_path << std::endl;
        // 清理临时片段文件和列表文件
        std::string cleanup_command = "rm " + output_dir_ + "/clip_*.avi " + output_dir_ + "/concat_list.txt";
        system(cleanup_command.c_str());
        return true;
    } else {
        std::cerr << "\n使用 FFmpeg 剪辑视频片段失败，错误代码: " << ffmpeg_result << std::endl;
        std::cerr << "请确保 FFmpeg 已安装并在系统路径中。" << std::endl;
        return false;
    }
}
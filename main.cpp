#include "video_io.h"
#include "face_detector.h"
#include "face_recognizer.h"
#include "target_tracker.h"
#include "vector_math.h"
#include "config.h"
#include "timing_logger.h"
#include "target_processing.h"
#include "video_clipper.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>

int main(int argc, char *argv[]) {
    std::string input_video_path;
    std::string target_face_path;
    std::string output_name = "output_tracked"; // 默认输出文件夹
    std::string output_tracked_path = output_name + "/output_tracked.avi";

    // 解析命令行参数
    if (argc == 5) {
        input_video_path = argv[1];
        target_face_path = argv[2];
        output_name = argv[3];
        float similarity_arg = std::stof(argv[4]);
        Config::setSimilarityThreshold(similarity_arg);
        output_tracked_path = output_name + "/output_tracked.avi";
    } else {
        std::cerr << "用法: " << argv[0] << "<输入视频路径> <目标图片路径> <输出文件夹> <相似度阈值>" << std::endl;
        std::cerr << "将使用默认配置: " << std::endl;
        std::cerr << "  输入视频路径: " << Config::VIDEO_INPUT_PATH << std::endl;
        std::cerr << "  目标图片路径: " << Config::TARGET_FACE_PATH << std::endl;
        std::cerr << "  输出跟踪视频: " << output_name << "/output_tracked.avi" << std::endl;
        std::cerr << "  输出剪辑视频: " << output_name << "/target_clips_.avi" << std::endl;
        std::cerr << "  相似度阈值: " << Config::SIMILARITY_THRESHOLD << std::endl;
        input_video_path = Config::VIDEO_INPUT_PATH;
        target_face_path = Config::TARGET_FACE_PATH;
        output_tracked_path = output_name + "/output_tracked.avi";
    }

    // 初始化视频输入/输出
    VideoIO video_io(input_video_path, output_tracked_path);

    std::string folderPath = output_name;
    video_io.createDirectory(folderPath); // 创建保存输出的文件夹
    if (!video_io.open()) {
        std::cerr << "无法打开视频文件: " << input_video_path << std::endl;
        return -1;
    }

    // 获取视频信息
    double fps = video_io.getFPS();
    cv::Size frame_size = video_io.getFrameSize();
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    long total_frames = video_io.getTotalFrames();
    std::string fourcc_str = video_io.getFourccString();
    double duration_seconds = video_io.getDuration();

    // 输出视频信息
    std::cout << "视频文件: " << input_video_path << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "帧率 (FPS): " << fps << std::endl;
    std::cout << "帧尺寸 (宽度 x 高度): " << frame_size.width << " x " << frame_size.height << std::endl;
    std::cout << "编码格式 (FourCC): " << fourcc_str << std::endl;
    std::cout << "总帧数: " << total_frames << std::endl;

    // 计算并输出视频总时长
    if (duration_seconds > 0) {
        int hours = static_cast<int>(duration_seconds / 3600);
        duration_seconds = std::fmod(duration_seconds, 3600.0);
        int minutes = static_cast<int>(duration_seconds / 60);
        double seconds = std::fmod(duration_seconds, 60.0);

        std::cout << "视频总时长: ";
        if (hours > 0) {
            std::cout << hours << " 小时 ";
        }
        if (minutes > 0 || hours > 0) {
            std::cout << minutes << " 分钟 ";
        }
        std::cout << std::fixed << std::setprecision(2) << seconds << " 秒" << std::endl;
    } else {
        std::cout << "视频总时长: 无法获取或计算。" << std::endl;
    }
    std::cout << "-----------------------------------" << std::endl;

    // 初始化视频写入器
    cv::VideoWriter writer(output_tracked_path, codec, fps, frame_size, true);
    if (!writer.isOpened()) {
        std::cerr << "无法打开输出跟踪视频文件: " << output_tracked_path << std::endl;
        return -1;
    }

    // 初始化目标处理类
    TargetProcessing target_processor(
        "models/yolov8n-face.onnx",
        "models/arcface.onnx",
        target_face_path,
        0.5f, // IOU阈值
        5 // 最大丢失帧数
    );

    cv::Mat frame;
    int frame_number = 0;
    TimingLogger total_timer;
    total_timer.start();

    bool is_end_early = false;

    // 逐帧处理视频
    while (video_io.readFrame(frame)) {
        if (!target_processor.processFrame(frame, frame_number)) {
            break;
        }
        cv::imshow("Video", frame);
        writer.write(frame);
        frame_number++;
        if (cv::waitKey(1) == 27) {
            is_end_early = true;
            // 在视频处理提前终止时调用 finalize 方法，传递最后一帧的帧号
            // 它的作用是记录目标出现的最后一帧，因为如果提前终止，而此时又能检测到目标，那么目标出现的最后一帧就没有被记录，此时要传入最后一帧来更新目标检测列表
            target_processor.finalize(frame_number - 1);
            break;
        }
    }

    if(!is_end_early) target_processor.finalize(frame_number - 1); // 在视频处理循环结束后调用 finalize 方法，传递最后一帧的帧号

    double total_time = total_timer.end();
    writer.release();
    video_io.release();
    cv::destroyAllWindows();

    // 打印耗时统计信息
    std::cout << "\n------------------- 耗时统计 -------------------" << std::endl;
    std::cout << "总处理帧数: " << frame_number << std::endl;
    std::cout << "总处理时间: " << std::fixed << std::setprecision(3) << total_time << " 秒" << std::endl;
    std::cout << "平均每帧处理时间: " << std::fixed << std::setprecision(3) << total_time / frame_number << " 秒" << std::endl;
    std::cout << "\n各模块耗时:" << std::endl;
    std::cout << "  - 人脸检测总耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalDetectionTime() << " 秒" << std::endl;
    std::cout << "    - 平均每帧耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalDetectionTime() / frame_number << " 秒" << std::endl;
    std::cout << "  - 目标跟踪总耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalTrackingTime() << " 秒" << std::endl;
    std::cout << "    - 平均每帧耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalTrackingTime() / frame_number << " 秒" << std::endl;
    std::cout << "  - 人脸识别总耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalRecognitionTime() << " 秒" << std::endl;
    std::cout << "    - 平均每帧耗时: " << std::fixed << std::setprecision(3) << target_processor.getTotalRecognitionTime() / frame_number << " 秒" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 初始化视频剪辑类并剪辑视频
    VideoClipper clipper(input_video_path, output_name);
    clipper.clipVideo(target_processor.getTargetClips(), fps, frame_size, codec);

    // 保存目标人物的标注框信息到文件
    std::string rois_filename = output_name + "/video_rois.txt";
    target_processor.saveTargetROIsToFile(rois_filename);

    video_io.copyFile(input_video_path, folderPath); // 将输入视频复制到输出目录
    video_io.copyFile(target_face_path, folderPath); // 将目标人物图片复制到输出目录

    return 0;
}
#include "video_io.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

VideoIO::VideoIO(const std::string& input_path, const std::string& output_path)
    : input_path_(input_path), output_path_(output_path){
}

VideoIO::~VideoIO() {
    // 释放视频捕获资源
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoIO::createDirectory(const std::string& folderPath) {
    std::filesystem::path path(folderPath);
    try {
        if (!std::filesystem::exists(path)) {
            if (std::filesystem::create_directories(path)) {
                std::cout << "成功创建文件夹: " << folderPath << std::endl;
                return true;
            } else {
                std::cerr << "无法创建文件夹: " << folderPath << std::endl;
                return false;
            }
        } else {
            std::cout << "文件夹已存在: " << folderPath << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "创建文件夹时发生错误: " << e.what() << std::endl;
        return false;
    }
}

bool VideoIO::copyFile(const std::string& sourceFilePath, const std::string& destinationFolder) {
    std::filesystem::path source_file(sourceFilePath);
    std::filesystem::path destination_folder_path(destinationFolder);
    std::filesystem::path destination_file = destination_folder_path / source_file.filename();

    try {
        if (std::filesystem::exists(source_file)) {
            if (std::filesystem::exists(destination_file)) {
                std::cerr << "目标文件夹中已存在同名文件: " << destination_file.string() << "，跳过复制。" << std::endl;
                return false;
            } else {
                std::filesystem::copy(source_file, destination_file);
                std::cout << "成功复制文件 '" << source_file.filename().string() << "' 到 '" << destinationFolder << "'" << std::endl;
                return true;
            }
        } else {
            std::cerr << "源文件不存在: " << sourceFilePath << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "复制文件时发生错误: " << e.what() << std::endl;
        return false;
    }
}


bool VideoIO::open() {
    // 检查是否已经打开
    if (cap_.isOpened()) {
        return true;
    }

    // 尝试打开视频
    cap_.open(input_path_);
    if (!cap_.isOpened()) {
        return false;
    }

    return true;
}

bool VideoIO::isOpened() const {
    return cap_.isOpened();
}

bool VideoIO::readFrame(cv::Mat& frame) {
    if (!cap_.isOpened()) {
        return false;
    }

    return cap_.read(frame);
}

double VideoIO::getFPS() const {
    return cap_.isOpened() ? cap_.get(cv::CAP_PROP_FPS) : 0.0;
}

cv::Size VideoIO::getFrameSize() const {
    if (!cap_.isOpened()) {
        return cv::Size(0, 0);
    }
    return cv::Size(
        static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
}

long VideoIO::getTotalFrames() const {
    return cap_.isOpened() ? static_cast<long>(cap_.get(cv::CAP_PROP_FRAME_COUNT)) : 0;
}

int VideoIO::getFourcc() const {
    return cap_.isOpened() ? static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC)) : 0;
}

double VideoIO::getDuration() const {
    if (!cap_.isOpened()) {
        return 0.0;
    }
    double fps = getFPS();
    long totalFrames = getTotalFrames();
    if (fps > 0 && totalFrames > 0) {
        return static_cast<double>(totalFrames) / fps;
    } else {
        return 0.0;
    }
}

std::string VideoIO::getFourccString() const {
    int fourccInt = getFourcc();
    char fourccChars[5];
    fourccChars[0] = static_cast<char>(fourccInt & 0xFF);
    fourccChars[1] = static_cast<char>((fourccInt >> 8) & 0xFF);
    fourccChars[2] = static_cast<char>((fourccInt >> 16) & 0xFF);
    fourccChars[3] = static_cast<char>((fourccInt >> 24) & 0xFF);
    fourccChars[4] = '\0';
    return std::string(fourccChars);
}


// bool VideoIO::setFramePosition(int frameNumber) {
//     if (!cap_.isOpened() || frameNumber < 0) {
//         return false;
//     }
//     return cap_.set(cv::CAP_PROP_POS_FRAMES, frameNumber);
// }

void VideoIO::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}
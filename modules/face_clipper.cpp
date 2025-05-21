#include "face_clipper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>

FaceClipper::FaceClipper() {}

std::vector<cv::Rect> FaceClipper::loadROIsFromFile(const std::string& rois_file_path) {
    std::vector<cv::Rect> rois;
    std::ifstream rois_file(rois_file_path);
    if (rois_file.is_open()) {
        int frame_id, x, y, width, height;
        while (rois_file >> frame_id >> x >> y >> width >> height) {
            rois.push_back(cv::Rect(x, y, width, height));
        }
        rois_file.close();
        std::cout << "成功加载 " << rois.size() << " 个标注框信息。" << std::endl;
    } else {
        std::cerr << "无法打开标注框信息文件: " << rois_file_path << std::endl;
    }
    return rois;
}

bool FaceClipper::clipFaces(const std::string& video_path,
                           const std::string& rois_file_path,
                           const std::string& output_dir) {
    cv::VideoCapture video_capture(video_path);
    if (!video_capture.isOpened()) {
        std::cerr << "无法打开视频文件: " << video_path << std::endl;
        return false;
    }

    std::vector<cv::Rect> rois = loadROIsFromFile(rois_file_path);
    if (rois.empty()) {
        std::cerr << "没有加载到有效的标注框信息，无法进行裁剪。" << std::endl;
        video_capture.release();
        return false;
    }

    // 创建输出目录，如果不存在
    if (mkdir(output_dir.c_str(), 0777) == -1) {
        if (errno != EEXIST) {
            std::cerr << "无法创建输出目录: " << output_dir << std::endl;
            video_capture.release();
            return false;
        }
    }

    cv::Mat frame;
    int face_count = 0;
    for (size_t i = 0; i < rois.size() && video_capture.read(frame); ++i) {
        cv::Rect roi = rois[i];
        if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
            roi.x + roi.width <= frame.cols && roi.y + roi.height <= frame.rows) {
            cv::Mat face_image = frame(roi).clone();
            std::stringstream filename_ss;
            filename_ss << output_dir << "/face_" << std::setw(6) << std::setfill('0') << i << ".jpg";
            std::string filename = filename_ss.str();
            cv::imwrite(filename, face_image);
            //std::cout << "保存裁剪面部图像: " << filename << std::endl;
            face_count++;
        } else {
            std::cerr << "警告: 第 " << i << " 个标注框无效，跳过裁剪。" << std::endl;
        }
    }

    video_capture.release();
    std::cout << "共保存 " << face_count << " 张面部图像到 " << output_dir << std::endl;
    return true;
}
#include "config.h"
#include <string>
#include <iostream>

// 静态成员变量的定义和初始化
const std::string Config::VIDEO_INPUT_PATH = "./data/input_video.mp4";
const std::string Config::TARGET_FACE_PATH = "./data/target_face.jpg";

float Config::SIMILARITY_THRESHOLD = 0.35f; // 人脸相似度，一般设置为0.35即可

void Config::setSimilarityThreshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) {
        Config::SIMILARITY_THRESHOLD = threshold;
        std::cout << "成功设置相似度阈值为: " << Config::SIMILARITY_THRESHOLD << std::endl;
    } else {
        std::cerr << "警告: 无效的相似度阈值 (" << threshold << ")，应在 [0.0, 1.0] 范围内。将保持当前值: " << Config::SIMILARITY_THRESHOLD << std::endl;
    }
}
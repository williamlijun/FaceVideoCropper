#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config {
public:
    // 默认的输入视频路径
    static const std::string VIDEO_INPUT_PATH;

    // 默认的目标图片路径
    static const std::string TARGET_FACE_PATH;

    // 人脸相似度阈值
    static float SIMILARITY_THRESHOLD;

    // 修改人脸相似度阈值
    static void setSimilarityThreshold(float threshold);
};

#endif // CONFIG_H
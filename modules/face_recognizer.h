#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "face_detector.h" // 引入 FaceBox 结构

// 定义人脸识别器
class FaceRecognizer {
public:
    FaceRecognizer(const std::string& model_path, const std::string& target_face_path);

    ~FaceRecognizer() = default;

    // 提取人脸特征向量
    std::vector<float> extractFeatures(const cv::Mat& face);

    // 计算两组特征向量的余弦相似度
    float computeSimilarity(const std::vector<float>& features1, const std::vector<float>& features2);

    // 识别人脸：比较检测人脸与目标人脸的相似度
    bool recognize(const cv::Mat& face, float threshold, float& similarity);

private:
    // 预处理人脸图像：调整为 112x112，转换为 RGB，归一化
    void preprocess(const cv::Mat& face, std::vector<float>& input_tensor);

    Ort::Env env_; // ONNXRuntime 环境
    Ort::Session session_{nullptr}; // ONNXRuntime 会话
    std::vector<const char*> input_names_; // 模型输入名称（ "data"）
    std::vector<const char*> output_names_; // 模型输出名称（ "1333"）
    std::vector<float> target_features_; // 目标人脸的特征向量（512 维）
    const int input_size_ = 112; // ArcFace 输入尺寸（112x112）
};

#endif // FACE_RECOGNIZER_H
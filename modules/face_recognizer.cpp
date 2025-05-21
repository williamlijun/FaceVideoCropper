#include "face_recognizer.h"
#include "face_detector.h" // 添加人脸检测头文件
#include <algorithm>
#include <cmath>
#include <stdexcept>

// 构造函数：初始化 ArcFace 模型并提取目标人脸特征
FaceRecognizer::FaceRecognizer(const std::string& model_path, const std::string& target_face_path)
    : env_(ORT_LOGGING_LEVEL_WARNING) {
    // 初始化 ONNXRuntime 会话
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4); // 4 线程
    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    // 设置输入/输出名称（根据 arcface.onnx 模型）
    // 可以运行 python/check_onnx.py 脚本查看模型的输入输出名称（记得修改模型路径）
    Ort::AllocatorWithDefaultOptions allocator;
    input_names_.push_back("data"); // 输入名称为 "data"
    output_names_.push_back("1333"); // 输出名称为 "1333"

    // 加载目标人脸图像
    cv::Mat target_face = cv::imread(target_face_path);
    if (target_face.empty()) {
        throw std::runtime_error("无法加载目标人脸图像: " + target_face_path);
    }

    // 初始化人脸检测器（使用 YOLOv8 模型）
    std::string detector_model_path = "models/yolov8n-face.onnx";
    FaceDetector detector(detector_model_path, 0.5f, 0.45f); // 置信度阈值 0.5

    // 检测目标人脸区域
    std::vector<FaceBox> faces = detector.detect(target_face);
    if (faces.empty()) {
        throw std::runtime_error("目标人脸图像中未检测到人脸: " + target_face_path);
    }

    // 选择置信度最高的人脸
    auto max_face = std::max_element(faces.begin(), faces.end(),
        [](const FaceBox& a, const FaceBox& b) { return a.score < b.score; });
    int x1 = std::max(0, static_cast<int>(max_face->x1));
    int y1 = std::max(0, static_cast<int>(max_face->y1));
    int x2 = std::min(target_face.cols, static_cast<int>(max_face->x2));
    int y2 = std::min(target_face.rows, static_cast<int>(max_face->y2));
    if (x2 <= x1 || y2 <= y1) {
        throw std::runtime_error("目标人脸区域无效: " + target_face_path);
    }

    // 裁剪人脸区域
    cv::Mat face_roi = target_face(cv::Rect(x1, y1, x2 - x1, y2 - y1));

    // 提取目标人脸特征
    target_features_ = extractFeatures(face_roi);
}

// 提取人脸特征向量
std::vector<float> FaceRecognizer::extractFeatures(const cv::Mat& face) {
    // 预处理人脸图像
    std::vector<float> input_tensor;
    preprocess(face, input_tensor);

    // 准备输入张量（形状 [1, 3, 112, 112]）
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    Ort::MemoryInfo memory_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

    // 运行推理
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor_, 1, output_names_.data(), 1);

    // 获取输出特征向量（512 维）
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t feature_size = output_shape[1]; // 输出为 [1, 512]
    std::vector<float> features(output_data, output_data + feature_size);

    // 归一化特征向量（L2 范数）
    float norm = 0.0f;
    for (float val : features) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& val : features) {
            val /= norm;
        }
    } else {
        // 避免除0，返回单位向量
        std::fill(features.begin(), features.end(), 0);
        features[0] = 1.0f;
    }

    return features;
}

// 计算余弦相似度
float FaceRecognizer::computeSimilarity(const std::vector<float>& features1, const std::vector<float>& features2) {
    if (features1.size() != features2.size()) {
        throw std::runtime_error("特征向量维度不匹配");
    }
    float dot_product = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i) {
        dot_product += features1[i] * features2[i]; // 计算两个向量的点积
    }
    return dot_product; // 余弦相似度（已归一化）特征向量已经过 L2 归一化，其点积等于它们的余弦相似度
}

// 识别人脸
bool FaceRecognizer::recognize(const cv::Mat& face, float threshold, float& similarity) {
    // 提取检测人脸的特征
    std::vector<float> features = extractFeatures(face);
    // 计算与目标人脸的相似度
    similarity = computeSimilarity(features, target_features_);
    // // 调试输出相似度
    // std::cout << "Similarity: " << similarity << std::endl;
    // 判断是否超过阈值
    return similarity > threshold;
}

// 预处理人脸图像
void FaceRecognizer::preprocess(const cv::Mat& face, std::vector<float>& input_tensor) {
    // 调整大小到 112x112
    cv::Mat resized;
    
    // // ==== 【新增】小人脸超分放大保护 ====
    // cv::Mat face_input = face;
    // if (face.cols < 112 || face.rows < 112) {
    //     cv::resize(face, face_input, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    // }
    // // ====================================

    cv::resize(face, resized, cv::Size(input_size_, input_size_));

    // 轻度锐化提升模糊人脸细节
    cv::Mat sharpened;
    cv::GaussianBlur(resized, sharpened, cv::Size(0, 0), 3);
    cv::addWeighted(resized, 1.5, sharpened, -0.5, 0, resized);
    // 后续处理基于 sharpened 图
    resized = sharpened;

    // 转换为 RGB，归一化并标准化（均值 0.5，标准差 0.5）
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB); // OpenCV 为 BGR，模型需要 RGB

    // 标准化：(x - mean) / std
    float_img = (float_img - 0.5) / 0.5; // ArcFace 常用标准化

    // 转换为 CHW 格式
    input_tensor.resize(3 * input_size_ * input_size_);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_size_; ++h) {
            for (int w = 0; w < input_size_; ++w) {
                input_tensor[index++] = float_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}
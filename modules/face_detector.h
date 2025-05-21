#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// 定义人脸边界框结构体
struct FaceBox {
    float x1, y1, x2, y2; // 边界框坐标 (左上角 x1,y1, 右下角 x2,y2)
    float score;          // 置信度
};

// 定义人脸检测器类
class FaceDetector {
public:
    FaceDetector(const std::string& model_path, float conf_threshold = 0.5f, float nms_threshold = 0.4f); // 置信度阈值 （默认0.5） 非极大值抑制 (NMS) 阈值，用于去除重叠的冗余检测框 (默认 0.4)
    ~FaceDetector() = default;

    // 检测人脸
    std::vector<FaceBox> detect(const cv::Mat& frame);

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& input_tensor); // 预处理
    std::vector<FaceBox> postprocess(const std::vector<float>& output_tensor, int img_width, int img_height); // 后处理函数：解析模型的输出张量，提取人脸边界框和置信度
    void nms(std::vector<FaceBox>& boxes, float nms_threshold); // 非极大值抑制 (NMS) 函数：去除重叠的冗余检测框

    Ort::Env env_; // ONNX Runtime 环境
    Ort::Session session_{nullptr}; // ONNX Runtime 会话，用于加载和运行模型
    std::vector<const char*> input_names_; // 模型输入名称向量
    std::vector<const char*> output_names_; // 模型输出名称向量

    float conf_threshold_; // 置信度阈值
    float nms_threshold_; // NMS 阈值
    const int input_size_ = 640; // YOLOv8 输入尺寸
};

#endif // FACE_DETECTOR_H
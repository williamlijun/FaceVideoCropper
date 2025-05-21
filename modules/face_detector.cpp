#include "face_detector.h"
#include "vector_math.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// 初始化，主要初始化一些数据以及初始化 ONNX Runtime 会话
FaceDetector::FaceDetector(const std::string& model_path, float conf_threshold, float nms_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING), conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) { // env_(ORT_LOGGING_LEVEL_WARNING) // 初始化 ONNX Runtime 环境，设置日志级别为 WARNING
    Ort::SessionOptions session_options; // 创建 ONNX Runtime 会话选项
    session_options.SetIntraOpNumThreads(4); // 设置推理时使用的线程数为 4
    session_ = Ort::Session(env_, model_path.c_str(), session_options); // 创建 ONNX Runtime 会话

    // 获取输入/输出名称（YOLOv8-face 的输入为 "images"，输出为 "output0"）
    Ort::AllocatorWithDefaultOptions allocator; // 创建默认的内存分配器
    // 可以运行 python/check_onnx.py 脚本查看模型的输入输出名称
    /*
    Inputs:
    images [1, 3, 640, 640]

    Outputs:
    output0 [1, 5, 8400]
    */
    input_names_.push_back("images"); // 将输入名称添加到输入名称向量中
    output_names_.push_back("output0"); // 将输出名称添加到输出名称向量中
}

// 对某一帧进行人脸检测
std::vector<FaceBox> FaceDetector::detect(const cv::Mat& frame) {
    // 预处理，将输入图像转换为模型所需的格式
    std::vector<float> input_tensor;
    preprocess(frame, input_tensor);

    // 准备输入张量
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_}; // 输入形状：[batch_size, channels, height, width] [1, 3, 640, 640] 与模型输入一致
    Ort::MemoryInfo memory_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault); // 指定在 CPU 上分配内存
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size()); // 创建输入向量

    // 运行推理，执行 ONNX 模型的前向传播
    auto output_tensors = session_.Run( Ort::RunOptions{nullptr}, // 运行选项，nullptr 表示使用默认选项
                                        input_names_.data(), // 输入名称数组的指针
                                        &input_tensor_, // 输入张量数组的指针
                                        1, // 输入张量的数量
                                        output_names_.data(), // 输出名称数组的指针
                                        1); // 输出张量的数量

    // 获取输出张量
    float* output_data = output_tensors[0].GetTensorMutableData<float>(); // 获取输出张量的数据指针
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // 获取输出张量的形状
    // 输出格式为 [1, 5, 8400]，其中 8400 是预测框的数量，5 表示 [cx, cy, w, h, confidence]，需要转置为 [1, 8400, 5]
    std::vector<float> output_tensor(5 * 8400);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8400; ++j) {
            output_tensor[j * 5 + i] = output_data[i * 8400 + j];
        }
    }
    return postprocess(output_tensor, frame.cols, frame.rows);
}

// 预处理输入向量
void FaceDetector::preprocess(const cv::Mat& frame, std::vector<float>& input_tensor) {
    // 调整大小到 640x640，保持宽高比
    cv::Mat resized;
    float scale = std::min(input_size_ / (float)frame.cols, input_size_ / (float)frame.rows); // 计算缩放比例，取宽度和高度缩放比例的较小值，以保证完整图像被缩放进目标尺寸
    int new_width = static_cast<int>(frame.cols * scale);
    int new_height = static_cast<int>(frame.rows * scale);
    cv::resize(frame, resized, cv::Size(new_width, new_height)); // 使用计算出的新尺寸进行图像缩放

    // 填充到 640x640
    cv::Mat padded = cv::Mat::zeros(input_size_, input_size_, CV_8UC3); // 创建一个黑色的 640x640 的图
    resized.copyTo(padded(cv::Rect(0, 0, new_width, new_height)));  // 将缩放后的图像复制到黑色图像的左上角，实现填充

    // 转换为 RGB，归一化到 [0, 1]
    cv::Mat float_img;
    padded.convertTo(float_img, CV_32FC3, 1.0 / 255.0); // 将图像数据类型转换为 float，并将像素值从 [0, 255] 归一化到 [0, 1]
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB); // 将 OpenCV 默认的 BGR 颜色空间转换为模型所需的 RGB 颜色空间

    // 转换为 CHW (Channels, Height, Width) 格式
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

// 处理输出向量
std::vector<FaceBox> FaceDetector::postprocess(const std::vector<float>& output_tensor, int img_width, int img_height) {
    std::vector<FaceBox> boxes;
    // 输出格式为 [1, 8400, 5]（cx, cy, w, h, score）
    int num_boxes = output_tensor.size() / 5; // 计算预测框的总数量 8400
    float scale = std::min(input_size_ / (float)img_width, input_size_ / (float)img_height); // 获取预处理时的缩放比例

    // 遍历每个预测框
    for (int i = 0; i < num_boxes; ++i) {
        float score = output_tensor[i * 5 + 4]; // 获取当前预测框的置信度
        if (score < conf_threshold_) continue; // 如果置信度低于阈值，则忽略该框

        // 获取中心点坐标、宽度和高度（相对于 640x640 的输入图像）
        float cx = output_tensor[i * 5 + 0];
        float cy = output_tensor[i * 5 + 1];
        float w = output_tensor[i * 5 + 2];
        float h = output_tensor[i * 5 + 3];

        // 将边界框坐标从中心点格式转换为左上角和右下角格式，并映射回原始图像尺寸
        FaceBox box;
        box.x1 = (cx - w / 2) / scale;
        box.y1 = (cy - h / 2) / scale;
        box.x2 = (cx + w / 2) / scale;
        box.y2 = (cy + h / 2) / scale;
        box.score = score;
        boxes.push_back(box); // 将处理后的边界框添加到结果向量中
    }

    // 应用非极大值抑制 (NMS) 来去除重叠的冗余框
    nms(boxes, nms_threshold_);
    return boxes; // 返回最终检测到的人脸边界框向量
}

// 非极大值抑制 (NMS) 函数的实现
void FaceDetector::nms(std::vector<FaceBox>& boxes, float nms_threshold) {
    // 按照置信度从高到低对检测框进行排序
    std::sort(boxes.begin(), boxes.end(), [](const FaceBox& a, const FaceBox& b) {
        return a.score > b.score;
    });

    std::vector<FaceBox> result;
    std::vector<bool> keep(boxes.size(), true); // 用于标记每个框是否保留

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i]) continue;
        result.push_back(boxes[i]); // 将当前框添加到结果中
        // 将所有与当前框的 IoU 大于阈值的后续框标记为不保留
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!keep[j]) continue;
            float iou = VectorMath::computeIoU(boxes[i], boxes[j]); // 计算两个框的 IoU（交并比）
            if (iou > nms_threshold) keep[j] = false;
        }
    }
    boxes = result;
}
#include "target_processing.h"
#include "vector_math.h" // 包含向量数学相关函数的头文件
#include "config.h"      // 包含配置文件相关的头文件
#include <iostream>      // 包含输入输出流的头文件
#include <sstream>      // 包含字符串流的头文件
#include <iomanip>      // 包含格式化输入输出的头文件

// 构造函数，初始化人脸检测器、识别器和跟踪器
TargetProcessing::TargetProcessing(const std::string& detector_model_path,
                                     const std::string& recognizer_model_path,
                                     const std::string& target_face_path,
                                     float tracker_iou_threshold,
                                     int tracker_max_lost_frames)
    : detector_(detector_model_path, 0.5f, 0.45f),
      recognizer_(recognizer_model_path, target_face_path),
      tracker_(tracker_iou_threshold, tracker_max_lost_frames) {}

// 处理单帧视频图像
bool TargetProcessing::processFrame(cv::Mat& frame, int frame_number) {
    // 运行人脸检测
    detection_timer_.start();
    std::vector<FaceBox> detections = detector_.detect(frame); // 使用人脸检测器检测当前帧中的人脸
    total_detection_time_ += detection_timer_.end();

    // 更新跟踪目标
    tracking_timer_.start();
    tracker_.update(detections, frame, tracked_targets_); // 使用目标跟踪器更新跟踪目标列表
    total_tracking_time_ += tracking_timer_.end();

    bool current_frame_has_target = false; // 标记当前帧是否包含目标人物

    // 如果当前帧识别到人脸，进行人脸识别与目标跟踪
    if (!detections.empty()) {
        recognition_timer_.start();
        // 找到置信度最高的人脸检测框
        auto max_det = *std::max_element(detections.begin(), detections.end(),
                                         [](const FaceBox& a, const FaceBox& b) { return a.score < b.score; });
        // 遍历当前跟踪到的目标
        for (auto& target : tracked_targets_) {
            // 只有在目标未丢失且人脸足够大时才进行识别
            if (target.lost_frames == 0 && target.last_roi.width >= 50 && target.last_roi.height >= 50) {
                // 计算跟踪框与最高置信度检测框的 IoU
                float iou = VectorMath::computeIoU(target.box, max_det);
                // 如果 IoU 大于阈值，则进行人脸识别
                if (iou > 0.5f) {
                    cv::Mat face_roi = frame(target.last_roi);
                    float similarity;
                    // 调用人脸识别器进行识别，并获取相似度
                    bool recognized = recognizer_.recognize(face_roi, Config::SIMILARITY_THRESHOLD, similarity);
                    target.similarity = similarity; // 存储相似度

                    // 如果识别成功且当前目标不是目标人物，则更新目标信息
                    if (recognized && !target.is_target) {
                        target.is_target = true;
                        target.features = recognizer_.extractFeatures(face_roi);
                        // 如果跟踪器还没有指定目标 ID，则设置当前目标的 ID
                        if (tracker_.getTargetId() == -1) {
                            tracker_.setTargetId(target.id);
                        }
                    }
                }
            }
            // 如果当前目标是目标人物且未丢失，则标记当前帧包含目标人物
            if (target.is_target && target.lost_frames == 0) {
                current_frame_has_target = true;
            }
        }
        total_recognition_time_ += recognition_timer_.end();
    }

    // 记录目标人物出现的片段
    if (current_frame_has_target && !target_detected_) {
        target_detected_ = true;
        target_start_frame_ = frame_number;
    } else if (!current_frame_has_target && target_detected_) {
        target_detected_ = false;
        target_clips_.push_back({target_start_frame_, frame_number - 1});
        target_start_frame_ = -1;
    }

    // 绘制跟踪框和相似度
    for (const auto& target : tracked_targets_) {
        if (target.lost_frames == 0) {
            // 如果是目标人物，框为绿色；否则为红色
            cv::Scalar color = target.is_target ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(frame, target.last_roi, color, 2); // 在帧上绘制矩形框
            if (target.is_target) {
                recordTargetROIInClip(target, frame_number); // 如果是目标人物，保存标注框信息 以及帧序号
            }
            // 显示跟踪 ID
            std::string id_str = "ID: " + std::to_string(target.id);
            cv::putText(frame, id_str,
                        cv::Point(target.last_roi.x, target.last_roi.y - 30), // 将 ID 显示在相似度上方
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            // 显示相似度 (保留四位小数)
            std::stringstream ss;
            // 防止出现6.40816e+29这样离谱的值，导致显示错误
            if(target.similarity > 0.0 && target.similarity <= 1.0) ss << std::fixed << std::setprecision(4) << target.similarity;
            else ss << std::fixed << "0.0000";
            cv::putText(frame, ss.str(),
                        cv::Point(target.last_roi.x, target.last_roi.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }

    return !frame.empty(); // 如果帧为空，则返回 false
}

// 获取当前跟踪到的目标列表
const std::vector<TrackedTarget>& TargetProcessing::getTrackedTargets() const {
    return tracked_targets_;
}

// 获取记录到的目标人物出现的时间片段列表
const std::vector<TargetClip>& TargetProcessing::getTargetClips() const {
    return target_clips_;
}

// 当视频运行到最后一帧或者提前终止时，获取当前的含有目标人物的最后出现位置
void TargetProcessing::finalize(int last_frame_number) {
    if (target_detected_) {
        target_clips_.push_back({target_start_frame_, last_frame_number});
        target_detected_ = false; // 重置标记
        target_start_frame_ = -1; // 重置起始帧
    }
}

// 获取人脸检测的总耗时
double TargetProcessing::getTotalDetectionTime() const {
    return total_detection_time_;
}

// 获取目标跟踪的总耗时
double TargetProcessing::getTotalTrackingTime() const {
    return total_tracking_time_;
}

// 获取人脸识别的总耗时
double TargetProcessing::getTotalRecognitionTime() const {
    return total_recognition_time_;
}

// 记录目标人物在当前片段中的 ROI 以及帧序号
void TargetProcessing::recordTargetROIInClip(const TrackedTarget& target, int& frame_number) {
    if (target.is_target && target.lost_frames == 0) {
        target_face_rois_in_clips_.push_back(target.last_roi);
        target_frame_indices_.push_back(frame_number);
    }
}

// 获取记录到的目标人物在当前片段中的 ROI 列表
const std::vector<cv::Rect>& TargetProcessing::getTargetROIsInClip() const {
    return target_face_rois_in_clips_;
}

// 将记录到的目标人物的 ROI 保存到文件中
void TargetProcessing::saveTargetROIsToFile(const std::string& filename) const {
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        for (size_t i = 0; i < target_face_rois_in_clips_.size(); ++i) {
            const auto& roi = target_face_rois_in_clips_[i];
            int frame_index = target_frame_indices_[i];
            outfile << frame_index << " " << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << std::endl;
        }
        outfile.close();
        std::cout << "目标 ROI 信息已保存到: " << filename << std::endl;
    } else {
        std::cerr << "无法打开文件以保存目标 ROI 信息: " << filename << std::endl;
    }
}
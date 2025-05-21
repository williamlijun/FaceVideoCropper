#include "target_tracker.h"
#include "vector_math.h"
#include <algorithm>

// 构造函数
// 初始化下一个分配的 ID 为 0
// 初始化目标 ID 为 -1，表示尚未设置
TargetTracker::TargetTracker(float iou_threshold, int max_lost_frames)
    : iou_threshold_(iou_threshold), max_lost_frames_(max_lost_frames),
      next_id_(0), target_id_(-1) {}

// 计算 IoU
float TargetTracker::computeIoU(const FaceBox& box1, const FaceBox& box2) {
    return VectorMath::computeIoU(box1, box2);
}

// 更新跟踪目标
void TargetTracker::update(const std::vector<FaceBox>& detections, const cv::Mat& frame,
                          std::vector<TrackedTarget>& tracked_targets) {
    std::vector<TrackedTarget> updated_targets; // 存储更新后的跟踪目标
    std::vector<bool> matched_detections(detections.size(), false); // 标记每个检测框是否已与现有目标匹配

    // 尝试将现有的跟踪目标与当前帧的检测结果进行匹配
    for (auto& target : tracked_targets) {
        float max_iou = 0.0f;
        int max_idx = -1;

        // 遍历当前帧的所有检测框，寻找与当前跟踪目标最匹配的检测框
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!matched_detections[i]) {
                float iou = computeIoU(target.box, detections[i]);
                if (iou > max_iou) {
                    max_iou = iou;
                    max_idx = i;
                }
            }
        }

        // 如果找到了一个匹配的检测框（IoU 大于阈值）
        if (max_idx != -1 && max_iou > iou_threshold_) {
            target.box = detections[max_idx]; // 更新跟踪目标的边界框为匹配的检测框
            // 更新上次成功跟踪到的人脸区域
            target.last_roi = cv::Rect(
                std::max(0, static_cast<int>(target.box.x1)),
                std::max(0, static_cast<int>(target.box.y1)),
                std::min(frame.cols, static_cast<int>(target.box.x2)) - std::max(0, static_cast<int>(target.box.x1)),
                std::min(frame.rows, static_cast<int>(target.box.y2)) - std::max(0, static_cast<int>(target.box.y1))
            );
            target.lost_frames = 0;
            matched_detections[max_idx] = true;
            updated_targets.push_back(target);
        } else {
            // 如果没有找到匹配的检测框，则增加该跟踪目标的丢失帧计数
            target.lost_frames++;
            if (target.lost_frames <= max_lost_frames_) {
                updated_targets.push_back(target);
            }
            // 超过最大丢失帧数的目标将不会被添加到 updated_targets，从而被认为丢失
        }
    }

    // 为未匹配的检测框创建新目标
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matched_detections[i]) {
            TrackedTarget new_target;
            new_target.box = detections[i];
            new_target.id = next_id_++;
            new_target.last_roi = cv::Rect(
                std::max(0, static_cast<int>(detections[i].x1)),
                std::max(0, static_cast<int>(detections[i].y1)),
                std::min(frame.cols, static_cast<int>(detections[i].x2)) - std::max(0, static_cast<int>(detections[i].x1)),
                std::min(frame.rows, static_cast<int>(detections[i].y2)) - std::max(0, static_cast<int>(detections[i].y1))
            );
            new_target.is_target = false;
            new_target.lost_frames = 0;
            updated_targets.push_back(new_target);
        }
    }

    // 更新跟踪目标列表
    tracked_targets = updated_targets;
}
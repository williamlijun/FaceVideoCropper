#ifndef TARGET_TRACKER_H
#define TARGET_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "face_detector.h"

// 跟踪目标结构
struct TrackedTarget {
    FaceBox box; // 当前帧的跟踪框（人脸边界框）
    int id; // 目标的唯一 ID，用于区分不同的跟踪目标
    float similarity; // 相似度
    cv::Rect last_roi; // 上一次成功跟踪到的人脸区域（OpenCV 的矩形表示）
    bool is_target; // 标记该目标是否为我们想要跟踪的目标人脸（通过人脸识别模块判断）
    int lost_frames; // 连续丢失跟踪的帧数，当超过一定阈值时，认为目标丢失
    std::vector<float> features; // 目标人脸特征（用于重新匹配）
};

// 目标跟踪器类 负责在视频帧序列中跟踪检测到的人脸目标
class TargetTracker {
public:
    // iou_threshold: IoU（交并比）阈值，用于匹配当前帧的检测结果和上一帧的跟踪目标 (默认 0.5)
    // max_lost_frames: 最大允许连续丢失跟踪的帧数，超过此阈值将认为目标丢失 (默认 5)
    TargetTracker(float iou_threshold = 0.5f, int max_lost_frames = 5);

    // 更新跟踪目标：接收当前帧的检测结果，更新现有的跟踪目标，并为新的检测结果创建新的跟踪目标
    void update(const std::vector<FaceBox>& detections, const cv::Mat& frame,
                std::vector<TrackedTarget>& tracked_targets);

    // 获取目标 ID
    int getTargetId() const { return target_id_; }
    // 设置目标 ID
    void setTargetId(int id) { target_id_ = id; }

private:
    float iou_threshold_; // IoU 阈值
    int max_lost_frames_; // 最大丢失帧数
    int next_id_; // 下一个分配的 ID
    int target_id_; // 目标人脸的固定 ID

    // 计算两个框的 IoU
    float computeIoU(const FaceBox& box1, const FaceBox& box2);
};

#endif // TARGET_TRACKER_H
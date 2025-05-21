#ifndef TARGET_PROCESSING_H
#define TARGET_PROCESSING_H

#include "video_io.h"        // 包含自定义的视频输入输出类的头文件
#include "face_detector.h"   // 包含人脸检测器类的头文件
#include "face_recognizer.h" // 包含人脸识别器类的头文件
#include "target_tracker.h"  // 包含目标跟踪器类的头文件
#include "timing_logger.h"   // 引入计时器类的头文件
#include <opencv2/opencv.hpp> // 包含 OpenCV 库的头文件
#include <vector>            // 包含动态数组容器的头文件
#include <iomanip>           // 包含格式化输入输出的头文件

// 存储目标人物出现的时间片段的结构体
struct TargetClip {
    int start_frame; // 目标人物出现的起始帧编号
    int end_frame;   // 目标人物出现的结束帧编号
};

// 目标处理类：负责视频帧的逐帧处理，包括检测、跟踪、识别和记录
class TargetProcessing {
public:
    // 构造函数
    TargetProcessing(const std::string& detector_model_path,    // 人脸检测模型路径
                     const std::string& recognizer_model_path,  // 人脸识别模型路径
                     const std::string& target_face_path,       // 目标人脸图像路径
                     float tracker_iou_threshold,              // 目标跟踪器的 IoU 阈值
                     int tracker_max_lost_frames);             // 目标跟踪器的最大丢失帧数

    // 处理单帧视频图像
    // 参数：
    //   frame: 当前帧的 OpenCV Mat 对象 (输入/输出，会绘制跟踪框)
    //   frame_number: 当前帧的编号
    // 返回值：
    //   bool: 如果帧处理成功（非空），则返回 true，否则返回 false
    bool processFrame(cv::Mat& frame, int frame_number);

    // 获取当前跟踪到的目标列表
    const std::vector<TrackedTarget>& getTrackedTargets() const;

    // 获取记录到的目标人物出现的时间片段列表
    const std::vector<TargetClip>& getTargetClips() const;

    // 获取人脸检测的总耗时
    double getTotalDetectionTime() const;

    // 获取目标跟踪的总耗时
    double getTotalTrackingTime() const;

    // 获取人脸识别的总耗时
    double getTotalRecognitionTime() const;

    // 当视频运行到最后一帧或者提前终止时，获取当前的含有目标人物的最后出现位置
    void finalize(int last_frame_number);

    void recordTargetROIInClip(const TrackedTarget& target, int& frame_number);
    const std::vector<cv::Rect>& getTargetROIsInClip() const;
    void saveTargetROIsToFile(const std::string& filename) const;

private:
    FaceDetector detector_;         // 人脸检测器对象
    FaceRecognizer recognizer_;     // 人脸识别器对象
    TargetTracker tracker_;        // 目标跟踪器对象
    TimingLogger detection_timer_;  // 人脸检测计时器
    TimingLogger tracking_timer_;   // 目标跟踪计时器
    TimingLogger recognition_timer_; // 人脸识别计时器
    double total_detection_time_ = 0.0;   // 累积的人脸检测总耗时
    double total_tracking_time_ = 0.0;    // 累积的目标跟踪总耗时
    double total_recognition_time_ = 0.0; // 累积的人脸识别总耗时
    std::vector<TrackedTarget> tracked_targets_; // 当前跟踪到的目标列表
    std::vector<TargetClip> target_clips_;     // 记录到的目标人物出现的时间片段列表
    bool target_detected_ = false;            // 标记当前是否检测到目标人物
    int target_start_frame_ = -1;             // 记录目标人物开始出现的帧编号
    std::vector<cv::Rect> target_face_rois_in_clips_;
    std::vector<int> target_frame_indices_;  // 每个 ROI 所在帧号
};

#endif // TARGET_PROCESSING_H
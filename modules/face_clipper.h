#ifndef FACE_CLIPPER_H
#define FACE_CLIPPER_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class FaceClipper {
public:
    FaceClipper();
    bool clipFaces(const std::string& video_path,
                   const std::string& rois_file_path,
                   const std::string& output_dir);

private:
    std::vector<cv::Rect> loadROIsFromFile(const std::string& rois_file_path);
};

#endif // FACE_CLIPPER_H
#include "face_clipper.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "用法: " << argv[0] << " <经过处理剪切好的视频路径（名称为target_clips.avi）> <标注框信息文件> [输出目录 (默认为 ./faces)]" << std::endl;
        return -1;
    }

    std::string input_video_path = argv[1];
    std::string rois_file_path = argv[2];
    std::string output_dir = (argc == 4) ? argv[3] : "./faces";

    FaceClipper face_clipper;
    if (face_clipper.clipFaces(input_video_path, rois_file_path, output_dir)) {
        std::cout << "面部裁剪完成。" << std::endl;
        return 0;
    } else {
        std::cerr << "面部裁剪失败。" << std::endl;
        return 1;
    }
}
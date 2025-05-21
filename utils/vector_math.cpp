#include "vector_math.h"
#include <algorithm>

namespace VectorMath {
    float computeIoU(const FaceBox& box1, const FaceBox& box2) {
        float x1 = std::max(box1.x1, box2.x1);
        float y1 = std::max(box1.y1, box2.y1);
        float x2 = std::min(box1.x2, box2.x2);
        float y2 = std::min(box1.y2, box2.y2);

        float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

        return inter_area / (area1 + area2 - inter_area);
    }
}
#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "face_detector.h"
#include <vector>

namespace VectorMath {
    float computeIoU(const FaceBox& box1, const FaceBox& box2);
}

#endif // VECTOR_MATH_H
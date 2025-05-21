#include "timing_logger.h"
#include <chrono>

TimingLogger::TimingLogger() : start_time_(std::chrono::high_resolution_clock::now()) {}

void TimingLogger::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

double TimingLogger::end() {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time_;
    return duration.count();
}
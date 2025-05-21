#ifndef TIMING_LOGGER_H
#define TIMING_LOGGER_H

#include <chrono>

class TimingLogger {
public:
    TimingLogger();
    void start();
    double end();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#endif // TIMING_LOGGER_H
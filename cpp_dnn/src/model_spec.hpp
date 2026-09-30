#pragma once
#include <array>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace yrd {

constexpr int kInputSize = 640;
constexpr int kNumAnchors = 5;
constexpr int kNumClasses = 3;
constexpr int kRawChannels = 37;
constexpr int kClsBase = 34;
constexpr float kXyRadius = 1.5f;
constexpr uchar kPadGray = 128;

struct StrideSpec {
    int stride;
    std::array<std::array<float, 2>, 5> anchors;
};

inline const std::vector<StrideSpec>& getStrideSpecs() {
    static const std::vector<StrideSpec> specs = {
        {32, {{{142.f, 110.f}, {167.f, 177.f}, {192.f, 243.f}, {326.f, 322.f}, {459.f, 401.f}}}},
        {16, {{{36.f, 75.f}, {56.f, 65.f}, {76.f, 55.f}, {74.f, 101.f}, {72.f, 146.f}}}},
        {8,  {{{12.f, 16.f}, {16.f, 26.f}, {19.f, 36.f}, {30.f, 32.f}, {40.f, 28.f}}}}
    };
    return specs;
}

inline const std::vector<std::string>& getClassNames() {
    static const std::vector<std::string> names = {"person", "vehicle", "other"};
    return names;
}

inline const std::vector<cv::Scalar>& getClassColorsBgr() {
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0)
    };
    return colors;
}

} // namespace yrd

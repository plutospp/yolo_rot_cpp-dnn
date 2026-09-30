#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include "model_spec.hpp"

namespace yrd {

struct Detection {
    int cls;
    float conf;
    float cx, cy;
    float W0, H0;
    float W45, H45;
    float a, b;
    float theta;
};

enum class Engine { New, Classic };

cv::Mat letterbox(const cv::Mat& bgr8uc3);
cv::Mat rotateCanvas(const cv::Mat& canvas, int angleDeg);

class YoloRotDetector {
public:
    YoloRotDetector(const std::string& onnxPath, Engine engine = Engine::New);

    std::vector<Detection> detect(const cv::Mat& canvasBgr, float confThr, float nmsIou, double* inferMs = nullptr);

    const std::string& outputShapes() const { return outputShapesStr_; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> outNames_;
    std::string outputShapesStr_;
};

} // namespace yrd

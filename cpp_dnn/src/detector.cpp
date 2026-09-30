#include "detector.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace yrd {

inline float sig(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

cv::Mat letterbox(const cv::Mat& bgr8uc3) {
    int cols = bgr8uc3.cols;
    int rows = bgr8uc3.rows;
    double scale = std::min(640.0 / cols, 640.0 / rows);
    int nw = static_cast<int>(cols * scale);
    int nh = static_cast<int>(rows * scale);

    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(kPadGray, kPadGray, kPadGray));
    cv::Rect roi((640 - nw) / 2, (640 - nh) / 2, nw, nh);

    if (nw == cols && nh == rows) {
        bgr8uc3.copyTo(canvas(roi));
    } else {
        cv::resize(bgr8uc3, canvas(roi), cv::Size(nw, nh), 0, 0, cv::INTER_CUBIC);
    }
    return canvas;
}

cv::Mat rotateCanvas(const cv::Mat& canvas, int angleDeg) {
    int a = ((angleDeg % 360) + 360) % 360;
    if (a == 0) {
        return canvas;
    }

    double cx = (canvas.cols - 1.0) / 2.0;
    double cy = (canvas.rows - 1.0) / 2.0;

    double c, s;
    if (a == 90) {
        c = 0.0; s = 1.0;
    } else if (a == 180) {
        c = -1.0; s = 0.0;
    } else if (a == 270) {
        c = 0.0; s = -1.0;
    } else {
        double rad = a * CV_PI / 180.0;
        c = std::cos(rad);
        s = std::sin(rad);
    }

    cv::Matx23d M(c, s, (1.0 - c) * cx - s * cy,
                 -s, c, s * cx + (1.0 - c) * cy);

    cv::Mat out;
    cv::warpAffine(canvas, out, M, canvas.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(kPadGray));
    return out;
}

static void makeDetection(int cls, float conf, float cx, float cy, float W0, float H0, float W45r, float H45r, std::vector<Detection>& cands) {
    double Sxx = static_cast<double>(W0) * W0;
    double Syy = static_cast<double>(H0) * H0;
    double Sxy = (static_cast<double>(W45r) * W45r - static_cast<double>(H45r) * H45r) / 2.0;

    double tr = (Sxx + Syy) / 2.0;
    double term = (Sxx - Syy) / 2.0;
    double r = std::sqrt(std::max(term * term + Sxy * Sxy, 0.0));

    double a = std::sqrt(std::max(tr + r, 1e-12));
    double b = std::sqrt(std::max(tr - r, 0.0));

    double th = 0.5 * std::atan2(2.0 * Sxy, Sxx - Syy);
    th += CV_PI / 2.0;
    th -= CV_PI * std::floor(th / CV_PI);
    th -= CV_PI / 2.0;

    double sum = (a * a + b * b) / 2.0;
    double diff = (a * a - b * b) / 2.0;
    double s2 = std::sin(2.0 * th);

    double W45 = std::sqrt(std::max(sum + diff * s2, 1e-12));
    double H45 = std::sqrt(std::max(sum - diff * s2, 1e-12));

    Detection det;
    det.cls = cls;
    det.conf = conf;
    det.cx = cx;
    det.cy = cy;
    det.W0 = W0;
    det.H0 = H0;
    det.W45 = static_cast<float>(W45);
    det.H45 = static_cast<float>(H45);
    det.a = static_cast<float>(a);
    det.b = static_cast<float>(b);
    det.theta = static_cast<float>(th);

    cands.push_back(det);
}

static const StrideSpec& specForGrid(int H) {
    int stride = 640 / H;
    for (const auto& spec : getStrideSpecs()) {
        if (spec.stride == stride) {
            return spec;
        }
    }
    throw std::runtime_error("unknown grid size " + std::to_string(H));
}

static void decodeStride(const cv::Mat& m, float confThr, std::vector<Detection>& cands) {
    int H = m.size[2];
    int W = m.size[3];
    const StrideSpec& spec = specForGrid(H);
    int s = spec.stride;

    const float* data = m.ptr<float>();
    int hw = H * W;

    for (int gy = 0; gy < H; ++gy) {
        for (int gx = 0; gx < W; ++gx) {
            int offset = gy * W + gx;

            int ci = 0;
            float max_cls_logit = data[kClsBase * hw + offset];
            for (int c = 1; c < kNumClasses; ++c) {
                float logit = data[(kClsBase + c) * hw + offset];
                if (logit > max_cls_logit) {
                    max_cls_logit = logit;
                    ci = c;
                }
            }
            float clsP = sig(max_cls_logit);
            if (clsP < confThr) continue;

            float tx0 = data[0 * hw + offset];
            float ty0 = data[1 * hw + offset];
            float cx = (3.0f * sig(tx0 / kXyRadius) - 1.0f + static_cast<float>(gx)) * static_cast<float>(s);
            float cy = (3.0f * sig(ty0 / kXyRadius) - 1.0f + static_cast<float>(gy)) * static_cast<float>(s);

            for (int i = 0; i < kNumAnchors; ++i) {
                float tobj0_i = data[(6 + 3 * i) * hw + offset];
                for (int j = 0; j < kNumAnchors; ++j) {
                    float tobj45_j = data[(21 + 3 * j) * hw + offset];
                    float conf = sig(std::min(tobj0_i, tobj45_j)) * clsP;
                    if (conf < confThr) continue;

                    float tw0_i = data[(4 + 3 * i) * hw + offset];
                    float th0_i = data[(5 + 3 * i) * hw + offset];
                    float sig_w0 = sig(tw0_i);
                    float sig_h0 = sig(th0_i);
                    float bw0 = (2.0f * sig_w0) * (2.0f * sig_w0) * spec.anchors[i][0];
                    float bh0 = (2.0f * sig_h0) * (2.0f * sig_h0) * spec.anchors[i][1];
                    if (bw0 < 2.0f || bh0 < 2.0f) continue;

                    float tw45_j = data[(19 + 3 * j) * hw + offset];
                    float th45_j = data[(20 + 3 * j) * hw + offset];
                    float sig_w45 = sig(tw45_j);
                    float sig_h45 = sig(th45_j);
                    float bw45 = (2.0f * sig_w45) * (2.0f * sig_w45) * spec.anchors[j][0];
                    float bh45 = (2.0f * sig_h45) * (2.0f * sig_h45) * spec.anchors[j][1];

                    makeDetection(ci, conf, cx, cy, bw0 / 2.0f, bh0 / 2.0f, bw45 / 2.0f, bh45 / 2.0f, cands);
                }
            }
        }
    }
}

YoloRotDetector::YoloRotDetector(const std::string& onnxPath, Engine engine) {
    net_ = cv::dnn::readNetFromONNX(onnxPath, engine == Engine::New ? cv::dnn::ENGINE_NEW : cv::dnn::ENGINE_CLASSIC);
    if (net_.empty()) {
        throw std::runtime_error("failed to load ONNX model: " + onnxPath);
    }
    outNames_ = net_.getUnconnectedOutLayersNames();

    cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(kPadGray, kPadGray, kPadGray));
    cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob, "images");
    std::vector<cv::Mat> outs;
    net_.forward(outs, outNames_);

    if (outs.size() != 3) {
        throw std::runtime_error("unexpected model outputs size: expected 3, got " + std::to_string(outs.size()));
    }

    std::ostringstream ss;
    for (size_t i = 0; i < outs.size(); ++i) {
        const auto& m = outs[i];
        if (m.dims != 4 || m.size[0] != 1 || m.size[1] != kRawChannels || m.size[2] != m.size[3]) {
            throw std::runtime_error("unexpected model outputs shape");
        }
        int H = m.size[2];
        int s = 640 / H;
        if (s != 32 && s != 16 && s != 8) {
            throw std::runtime_error("unexpected model outputs stride: " + std::to_string(s));
        }
        ss << (i == 0 ? "" : " ") << "[" << m.size[0] << "," << m.size[1] << "," << m.size[2] << "," << m.size[3] << "]";
    }
    outputShapesStr_ = ss.str();
}

std::vector<Detection> YoloRotDetector::detect(const cv::Mat& canvasBgr, float confThr, float nmsIou, double* inferMs) {
    CV_Assert(canvasBgr.cols == 640 && canvasBgr.rows == 640 && canvasBgr.type() == CV_8UC3);

    cv::Mat blob = cv::dnn::blobFromImage(canvasBgr, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob, "images");

    std::vector<cv::Mat> outs;
    int64 t0 = cv::getTickCount();
    net_.forward(outs, outNames_);
    int64 t1 = cv::getTickCount();

    if (inferMs) {
        *inferMs = (t1 - t0) * 1000.0 / cv::getTickFrequency();
    }

    std::vector<Detection> cands;
    for (auto& out : outs) {
        cv::Mat m = out.isContinuous() ? out : out.clone();
        decodeStride(m, confThr, cands);
    }

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    boxes.reserve(cands.size());
    scores.reserve(cands.size());
    for (const auto& d : cands) {
        boxes.emplace_back(d.cx - d.W0, d.cy - d.H0, 2.0 * d.W0, 2.0 * d.H0);
        scores.push_back(d.conf);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, nmsIou, keep);

    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        result.push_back(cands[idx]);
    }
    return result;
}

} // namespace yrd

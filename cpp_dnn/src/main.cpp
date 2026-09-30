#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <cctype>
#include <algorithm>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "detector.hpp"

class LatestFrameGrabber {
public:
    LatestFrameGrabber(cv::VideoCapture& cap)
        : cap_(cap), seq_(0), stop_(false), failed_(false) {
        thread_ = std::thread(&LatestFrameGrabber::loop, this);
    }

    ~LatestFrameGrabber() {
        stop_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool latest(cv::Mat& frame, long long& seq) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_.empty()) return false;
        frame_.copyTo(frame);
        seq = seq_;
        return true;
    }

    bool failed() const { return failed_; }

private:
    void loop() {
        while (!stop_) {
            cv::Mat tmp;
            if (!cap_.read(tmp) || tmp.empty()) {
                failed_ = true;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                tmp.copyTo(frame_);
                seq_++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    cv::VideoCapture& cap_;
    std::mutex mutex_;
    cv::Mat frame_;
    long long seq_ = 0;
    std::atomic<bool> stop_;
    std::atomic<bool> failed_;
    std::thread thread_;
};

static bool isAllDigits(const std::string& str) {
    return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
}

static bool isImageFile(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tif" || ext == ".tiff" || ext == ".webp");
}

static void drawDashedLine(cv::Mat& img, cv::Point2f p1, cv::Point2f p2, const cv::Scalar& color, int dashLen = 10, int gapLen = 6) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist <= 1e-3f) return;

    float ux = dx / dist;
    float uy = dy / dist;

    float curr = 0.0f;
    while (curr < dist) {
        float next_dash = std::min(curr + dashLen, dist);
        cv::Point2f startPt(p1.x + ux * curr, p1.y + uy * curr);
        cv::Point2f endPt(p1.x + ux * next_dash, p1.y + uy * next_dash);
        cv::line(img, startPt, endPt, color, 1, cv::LINE_AA);
        curr = next_dash + gapLen;
    }
}

static cv::Mat render(const cv::Mat& rotatedCanvas, const std::vector<yrd::Detection>& dets,
                     int angle, double inferMs, double fps, bool paused) {
    cv::Mat canvas = rotatedCanvas.clone();

    for (const auto& d : dets) {
        cv::Scalar color = yrd::getClassColorsBgr()[d.cls % 3];
        cv::Scalar ink(255 - color[0], 255 - color[1], 255 - color[2]);

        // 1. Ellipse (72 points)
        std::vector<cv::Point> pts;
        pts.reserve(72);
        double cos_th = std::cos(d.theta);
        double sin_th = std::sin(d.theta);
        for (int k = 0; k < 72; ++k) {
            double tk = 2.0 * CV_PI * k / 72.0;
            double cos_t = std::cos(tk);
            double sin_t = std::sin(tk);
            double x = d.cx + d.a * cos_t * cos_th - d.b * sin_t * sin_th;
            double y = d.cy + d.a * cos_t * sin_th + d.b * sin_t * cos_th;
            pts.emplace_back(cvRound(x), cvRound(y));
        }
        std::vector<std::vector<cv::Point>> polyPts = { pts };
        cv::polylines(canvas, polyPts, true, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::polylines(canvas, polyPts, true, color, 2, cv::LINE_AA);

        // 2. 0° box
        cv::Point p1(cvRound(d.cx - d.W0), cvRound(d.cy - d.H0));
        cv::Point p2(cvRound(d.cx + d.W0), cvRound(d.cy + d.H0));
        cv::rectangle(canvas, p1, p2, color, 1);

        // 3. 45° box (dashed)
        constexpr double kSqrtHalf = 0.7071067811865475; // sqrt(0.5)
        std::array<std::pair<float, float>, 4> localCorners = {{{-d.W45, -d.H45}, {d.W45, -d.H45}, {d.W45, d.H45}, {-d.W45, d.H45}}};
        std::array<cv::Point2f, 4> worldCorners;
        for (int c = 0; c < 4; ++c) {
            float lx = localCorners[c].first;
            float ly = localCorners[c].second;
            float wx = d.cx + static_cast<float>((lx - ly) * kSqrtHalf);
            float wy = d.cy + static_cast<float>((lx + ly) * kSqrtHalf);
            worldCorners[c] = cv::Point2f(wx, wy);
        }
        for (int c = 0; c < 4; ++c) {
            drawDashedLine(canvas, worldCorners[c], worldCorners[(c + 1) % 4], color, 10, 6);
        }

        // 4. Label
        char labelBuf[64];
        std::snprintf(labelBuf, sizeof(labelBuf), "%s %.2f", yrd::getClassNames()[d.cls % 3].c_str(), d.conf);
        int base = 0;
        cv::Size ts = cv::getTextSize(labelBuf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);

        int x1 = cvRound(d.cx - d.W0);
        int y1 = std::max(cvRound(d.cy - d.H0), ts.height + base + 4);

        cv::rectangle(canvas, cv::Point(x1, y1 - ts.height - base - 4), cv::Point(x1 + ts.width + 4, y1), color, cv::FILLED);
        cv::putText(canvas, labelBuf, cv::Point(x1 + 2, y1 - base - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv::LINE_AA);
    }

    // 5. HUD bar below canvas
    cv::Mat hudBar(48, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    char line1[128];
    std::snprintf(line1, sizeof(line1), "angle %d deg CCW | infer %.0f ms | %.1f FPS | dets %zu%s",
                  angle, inferMs, fps, dets.size(), paused ? " | PAUSED" : "");
    const char* line2 = "a/d +-5  A/D +-45  r reset  space pause  q/Esc quit";

    cv::putText(hudBar, line1, cv::Point(10, 19), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(hudBar, line2, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::Mat out;
    cv::vconcat(canvas, hudBar, out);
    return out;
}

int main(int argc, char** argv) {
    const char* keys =
        "{help h usage ? |      | print help message}"
        "{model          |<none>| path to ONNX model file}"
        "{source         |<none>| video file, image file, or camera index}"
        "{conf           |0.5   | confidence threshold (0,1)}"
        "{nms            |0.6   | NMS IOU threshold (0,1]}"
        "{angle          |0     | network input CCW rotation in degrees}"
        "{engine         |new   | DNN engine: 'new' or 'classic'}"
        "{max_frames     |0     | max inference runs before exiting (0=no limit)}"
        "{headless       |      | run in headless mode without GUI}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("YOLO-rot C++ OpenCV-DNN Inference");

    if (parser.has("help") || !parser.has("model") || !parser.has("source") ||
        parser.get<std::string>("model") == "<none>" || parser.get<std::string>("source") == "<none>") {
        parser.printMessage();
        return 0;
    }

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    std::string modelPath = parser.get<std::string>("model");
    std::string sourceStr = parser.get<std::string>("source");
    float conf = parser.get<float>("conf");
    float nms = parser.get<float>("nms");
    int angle = parser.get<int>("angle");
    std::string engineStr = parser.get<std::string>("engine");
    int maxFrames = parser.get<int>("max_frames");
    bool headless = parser.has("headless");

    if (conf <= 0.0f || conf >= 1.0f) {
        std::cerr << "error: conf threshold must be in (0,1)" << std::endl;
        return 1;
    }
    if (nms <= 0.0f || nms > 1.0f) {
        std::cerr << "error: nms threshold must be in (0,1]" << std::endl;
        return 1;
    }
    if (maxFrames < 0) {
        std::cerr << "error: max_frames must be >= 0" << std::endl;
        return 1;
    }

    yrd::Engine engine;
    if (engineStr == "new") {
        engine = yrd::Engine::New;
    } else if (engineStr == "classic") {
        engine = yrd::Engine::Classic;
    } else {
        std::cerr << "error: invalid engine '" << engineStr << "'; expected 'new' or 'classic'" << std::endl;
        return 1;
    }

    angle = ((angle % 360) + 360) % 360;

    try {
        int64 t0 = cv::getTickCount();
        yrd::YoloRotDetector det(modelPath, engine);
        int64 t1 = cv::getTickCount();
        double loadMs = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        std::cerr << "loaded " << modelPath << " (engine " << engineStr << ") in "
                  << static_cast<int>(loadMs) << " ms; outputs " << det.outputShapes() << std::endl;

        bool isCam = isAllDigits(sourceStr);
        bool isImg = !isCam && isImageFile(sourceStr);

        cv::VideoCapture cap;
        cv::Mat imageFrame;
        std::unique_ptr<LatestFrameGrabber> grabber;

        if (isCam) {
            int camIdx = std::stoi(sourceStr);
            cap.open(camIdx);
            if (!cap.isOpened()) {
                std::cerr << "error: cannot open camera " << camIdx << std::endl;
                return 1;
            }
            grabber = std::make_unique<LatestFrameGrabber>(cap);
        } else if (isImg) {
            imageFrame = cv::imread(sourceStr, cv::IMREAD_COLOR);
            if (imageFrame.empty()) {
                std::cerr << "error: cannot read image: " << sourceStr << std::endl;
                return 1;
            }
        } else {
            cap.open(sourceStr);
            if (!cap.isOpened()) {
                std::cerr << "error: cannot open video source: " << sourceStr << std::endl;
                return 1;
            }
        }

        std::string winName = "yolo_rot_dnn";
        if (!headless) {
            cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
            cv::createTrackbar("angle", winName, nullptr, 359);
            cv::setTrackbarPos("angle", winName, angle);
        }

        bool paused = false;
        int runs = 0;
        long long frameId = -1;
        long long lastFrameId = -2;
        int lastAngle = -1;
        cv::Mat currentRawFrame;
        double fpsEma = 0.0;
        auto lastInferTime = std::chrono::steady_clock::now();

        while (true) {
            if (isCam && grabber->failed()) {
                std::cerr << "error: camera stopped delivering frames" << std::endl;
                return 1;
            }

            if (isImg) {
                currentRawFrame = imageFrame;
                frameId = 0;
            } else if (isCam) {
                if (!paused || currentRawFrame.empty()) {
                    long long seq = 0;
                    if (!grabber->latest(currentRawFrame, seq)) {
                        if (headless) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        } else {
                            cv::waitKey(1);
                        }
                        continue;
                    }
                    frameId = seq - 1;
                }
            } else { // Video
                if (!paused || currentRawFrame.empty()) {
                    cv::Mat tmp;
                    if (!cap.read(tmp) || tmp.empty()) {
                        std::cerr << "end of stream" << std::endl;
                        break;
                    }
                    currentRawFrame = tmp;
                    frameId++;
                }
            }

            if (currentRawFrame.channels() == 1) {
                cv::cvtColor(currentRawFrame, currentRawFrame, cv::COLOR_GRAY2BGR);
            } else if (currentRawFrame.channels() == 4) {
                cv::cvtColor(currentRawFrame, currentRawFrame, cv::COLOR_BGRA2BGR);
            }

            if (!headless) {
                angle = cv::getTrackbarPos("angle", winName);
                angle = ((angle % 360) + 360) % 360;
            }

            if (frameId != lastFrameId || angle != lastAngle) {
                cv::Mat canvas = yrd::letterbox(currentRawFrame);
                cv::Mat rot = yrd::rotateCanvas(canvas, angle);

                double inferMs = 0.0;
                auto now = std::chrono::steady_clock::now();
                std::vector<yrd::Detection> dets = det.detect(rot, conf, nms, &inferMs);
                runs++;

                double dtMs = std::chrono::duration<double, std::milli>(now - lastInferTime).count();
                lastInferTime = now;
                if (runs == 1) {
                    fpsEma = (inferMs > 0 ? 1000.0 / inferMs : 0.0);
                } else if (dtMs > 0) {
                    fpsEma = 0.8 * fpsEma + 0.2 * (1000.0 / dtMs);
                }

                if (headless) {
                    std::cout << "frame=" << frameId << " angle=" << angle << " infer_ms="
                              << std::fixed << std::setprecision(1) << inferMs << " dets=" << dets.size() << std::endl;
                    for (const auto& d : dets) {
                        std::cout << "det cls=" << yrd::getClassNames()[d.cls % 3]
                                  << " conf=" << std::setprecision(4) << d.conf
                                  << " cx=" << std::setprecision(2) << d.cx
                                  << " cy=" << d.cy
                                  << " W0=" << d.W0
                                  << " H0=" << d.H0
                                  << " W45=" << d.W45
                                  << " H45=" << d.H45
                                  << " a=" << d.a
                                  << " b=" << d.b
                                  << " theta=" << d.theta << std::endl;
                    }
                    std::fflush(stdout);
                } else {
                    cv::Mat vis = render(rot, dets, angle, inferMs, fpsEma, paused);
                    cv::imshow(winName, vis);
                }

                lastFrameId = frameId;
                lastAngle = angle;

                if (maxFrames > 0 && runs >= maxFrames) {
                    break;
                }
                if (headless && isImg) {
                    break;
                }
            }

            if (headless) {
                continue;
            }

            int waitMs = (paused || isImg) ? 30 : 1;
            int key = cv::waitKey(waitMs);
            if (cv::getWindowProperty(winName, cv::WND_PROP_VISIBLE) < 1) {
                break;
            }

            if (key > 0) {
                int k = key & 0xFF;
                if (k == 'q' || k == 27) {
                    break;
                } else if (k == 'a') {
                    angle = (angle + 5) % 360;
                    cv::setTrackbarPos("angle", winName, angle);
                } else if (k == 'd') {
                    angle = (angle - 5 + 360) % 360;
                    cv::setTrackbarPos("angle", winName, angle);
                } else if (k == 'A') {
                    angle = (angle + 45) % 360;
                    cv::setTrackbarPos("angle", winName, angle);
                } else if (k == 'D') {
                    angle = (angle - 45 + 360) % 360;
                    cv::setTrackbarPos("angle", winName, angle);
                } else if (k == 'r') {
                    angle = 0;
                    cv::setTrackbarPos("angle", winName, angle);
                } else if (k == ' ' && !isImg) {
                    paused = !paused;
                }
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

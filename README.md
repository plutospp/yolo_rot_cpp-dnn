# YOLO-rot C++ OpenCV-DNN Inference

A high-performance C++ inference application for **YOLO-rot** oriented bounding box (OBB) object detection running on OpenCV's DNN module (`cv::dnn`). It supports video files, USB camera streams, and still images, offering live rotation control during inference.

---

## Features

- **OpenCV 5 DNN Integration:** Runs exported ONNX models via `cv::dnn::readNetFromONNX` using either the new DNN engine or classic engine.
- **Multi-Input Sources:** Supports video files, USB camera indices (with real-time thread-safe frame grabbing to prevent buffer lag), and still images.
- **Live Rotation Control:** Real-time trackbar and hotkeys to rotate input frames dynamically while inference is running.
- **Visualization:**
  - Fitted oriented ellipse ($\mathbf{a}, \mathbf{b}, \theta$).
  - 0° axis-aligned bounding box.
  - 45° rotated dashed bounding box.
  - Class label with confidence score.
  - Real-time HUD showing rotation angle, inference time (ms), FPS, detection count, and pause status.
- **Headless Mode:** CLI-only output mode suitable for batch execution, testing, and parity verification.

---

## Project Structure

```
cpp_dnn/
├── export_onnx.py          # Python ONNX exporter with spec guard & graph optimizations
├── CMakeLists.txt          # CMake build configuration (C++17, OpenCV 5, MSVC settings)
└── src/
    ├── model_spec.hpp      # Anchor specifications, class names, and color constants
    ├── detector.hpp        # YoloRotDetector class declaration and utility signatures
    ├── detector.cpp        # Letterboxing, rotation, DNN forward pass, decoding, and NMS
    └── main.cpp            # CLI parser, LatestFrameGrabber, GUI/rendering, and main loop
```

---

## 1. Exporting the ONNX Model

Export the PyTorch checkpoint to a raw-head ONNX file compatible with OpenCV DNN:

```bash
python cpp_dnn/export_onnx.py \
    --config configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json \
    --weights best \
    --out model_data/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls_rawhead.onnx
```

### Exporter CLI Arguments
- `--config`: Model JSON config file (default: `configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json`).
- `--weights`: Weight checkpoint selector: `best`, `last_epoch`, or `last_batch` (default: `best`).
- `--out`: Destination path for output `.onnx` file.

---

## 2. Building the C++ Application

### Prerequisites
- C++17 compatible compiler (MSVC 19.4x / Visual Studio 2022 / VS Build Tools).
- CMake 3.21 or higher.
- Ninja build tool.
- OpenCV 5.0.0 (Windows prebuilt pack or built from source).

### Build Commands (Windows MSVC)

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S cpp_dnn -B cpp_dnn\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/david/opencv-5.0.0/opencv/build
cmake --build cpp_dnn\build
```

The compiled binary will be generated at `cpp_dnn\build\yolo_rot_dnn.exe` alongside necessary OpenCV DLLs.

---

## 3. Usage & CLI Options

Run the binary specifying `--model` and `--source`:

```cmd
cpp_dnn\build\yolo_rot_dnn.exe --model=model_data\rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls_rawhead.onnx --source=video.mp4
```

### CLI Arguments Table

| Option | Default | Description |
|---|---|---|
| `--model` | `<required>` | Path to the exported `_rawhead.onnx` model file. |
| `--source` | `<required>` | Input source: numeric index for USB camera (e.g. `0`), image file (`.png`, `.jpg`, etc.), or video file. |
| `--conf` | `0.5` | Confidence threshold in range `(0, 1)`. |
| `--nms` | `0.6` | Non-Maximum Suppression IoU threshold in range `(0, 1]`. |
| `--angle` | `0` | Initial input counter-clockwise rotation angle in degrees `[0, 360)`. |
| `--engine` | `new` | OpenCV DNN engine mode: `new` (default) or `classic`. |
| `--max_frames` | `0` | Maximum inference runs before exiting (`0` for unlimited). |
| `--headless` | `false` | Enable headless mode (no GUI window opened; prints detection text to stdout). |

---

## 4. Interactive Controls & UI

In GUI mode, a window displays the rotated 640×640 input canvas along with an angle trackbar and bottom HUD status bar.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `a` | Rotate input counter-clockwise by **+5°** |
| `d` | Rotate input clockwise by **-5°** |
| `A` | Rotate input counter-clockwise by **+45°** |
| `D` | Rotate input clockwise by **-45°** |
| `r` | Reset rotation angle to **0°** |
| `Space` | Toggle pause / resume (allows rotating frozen video frames) |
| `q` / `Esc` | Quit application |

---

## 5. Output Format in Headless Mode

When running with `--headless`, detection results are printed to stdout:

```text
frame=0 angle=0 infer_ms=28.4 dets=2
det cls=person conf=0.8851 cx=320.50 cy=240.10 W0=18.50 H0=42.00 W45=25.10 H45=35.20 a=43.10 b=17.80 theta=0.1250
det cls=vehicle conf=0.7912 cx=150.20 cy=380.40 W0=35.00 H0=22.50 W45=30.10 H45=28.40 a=36.20 b=21.80 theta=-0.3421
```

---

## License

Internal project module.

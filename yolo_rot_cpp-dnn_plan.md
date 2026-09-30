# YOLO-rot C++ OpenCV-DNN inference: video files and USB camera, with live rotation control

## Context
Build a C++ inference app that runs on OpenCV's DNN module (`cv::dnn`) for the yolo_rot model that is training right now. That model uses config `configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json` (classes: person, vehicle, other). Its checkpoints are in `logs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls/`.

- **Inputs:** a video file, a USB camera index, or a still image (still images are only there to make testing deterministic).
- **Live control:** while inference runs, the user sets the rotation of the network input with a trackbar and keys. Detections are drawn on the rotated frame: the 0° box, the 45° box, the ellipse and a label.
- **End state:** a `cpp_dnn/` folder containing an ONNX export script and a CMake C++ app. The app's detections must match the project's Python pipeline.

**Self-containment.** Every layout, constant and formula the implementation needs is written out in this plan. Repo code comes into play in exactly three ways, and none of it has to be read:
- **Imported by the export script:** the model class and the config loader. Exporting is impossible without the model definition.
- **Imported by the parity check:** the Python inference class, used as the numeric oracle.
- **Loaded as inputs:** the config, the checkpoint, and `img/street.jpg` for test data.

Nothing in the repo is modified.

**Environment (confirmed on this machine):**
- **OS:** Windows x64.
- **Compiler and build tools:** VS 2026 Build Tools with MSVC 14.51. CMake 4.3 and Ninja ship inside it: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\...`, with the environment set up by `VC\Auxiliary\Build\vcvars64.bat`.
- **Python:** `C:\Python313\python.exe`, with torch 2.11 (cu128), onnx 1.22, onnxscript 0.7, onnxruntime 1.27, Pillow 12.2 and opencv-python-headless 4.13 in the user site-packages.
- **Tools:** 7-Zip at `C:\Program Files\7-Zip\7z.exe`.
- **Not installed:** no C++ OpenCV.
- **GPU:** busy with training, so all work here runs on the CPU.

## Approach
Step 1 (Python export) is independent of Steps 2–4 (C++). Build the C++ app once Step 4 is written.

### Step 1 — Export script `cpp_dnn/export_onnx.py` (new file)
This script produces a "raw-head" ONNX file that OpenCV can run. Runtime prerequisites:
- Run it with `C:\Python313\python.exe` from any working directory.
- The script itself sets `REPO_ROOT = Path(__file__).resolve().parent.parent`, runs `sys.path.insert(0, str(REPO_ROOT))`, then calls `os.chdir(REPO_ROOT)`. Config paths inside the JSON are relative to the repo root.

1. **Imports.**
   - Standard library and packages: `argparse, hashlib, io, os, sys, pathlib.Path, numpy as np, onnx, onnxruntime as ort, torch, torch.nn as nn`.
   - From the repo: `from nets.yolo_rot import YoloRotBody, MaskedConv2d` and `from utils.model_config import load_model_config`.
2. **CLI.**
   - `--config`: default `configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json`.
   - `--weights`: one of `best`, `last_epoch`, `last_batch`; default `best`. If `best` is chosen and the file is missing, fall back to `last_epoch`.
   - `--out`: default `model_data/<cfg.name>_rawhead.onnx`, which resolves to `model_data/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls_rawhead.onnx`. The `_rawhead` suffix keeps it apart from other ONNX exports of this config, which use the expanded 228-channel layout.
   - Checkpoint path: `cfg.weights(which)`, i.e. `<save_dir>/best_epoch_weights.pth` or `<save_dir>/last_epoch_weights.pth` or `<save_dir>/last_batch_weights.pth`.
3. **Spec guard.** Define module constants that mirror `model_spec.hpp` exactly:
   - `SPEC_INPUT=[640,640]`
   - `SPEC_STRIDES=[32,16,8]`
   - `SPEC_MASK=[[10,11,12,13,14],[5,6,7,8,9],[0,1,2,3,4]]`
   - `SPEC_ANCHORS=[(12,16),(16,26),(19,36),(30,32),(40,28),(36,75),(56,65),(76,55),(74,101),(72,146),(142,110),(167,177),(192,243),(326,322),(459,401)]`
   - `SPEC_CLASSES=["person","vehicle","other"]`
   - `SPEC_XY_RADIUS=1.5`

   After `cfg = load_model_config(args.config)`, check every one of the following and `sys.exit(...)` on the first mismatch:
   - `cfg.head == "dual_bbox"`, and `cfg.shared_anchor`, `cfg.shared_xy`, `cfg.shared_cls` and `cfg.raw_head` are all true.
   - `cfg.cls_loss_type == "bce"`.
   - `cfg.input_shape`, `cfg.predict_strides` and `cfg.anchors_mask` equal the constants above.
   - The anchors equal `SPEC_ANCHORS`. Parse them from `cfg.anchors_path`: first line, split on `,`, convert to float, pair up.
   - The stripped lines of `cfg.classes_path` equal `SPEC_CLASSES`.
   - `cfg.positive_grid_radius + 0.5 == SPEC_XY_RADIUS`.

   Error text: `config <path>: <field>=<value> does not match cpp_dnn/src/model_spec.hpp (<expected>); update both together`.
4. **Load the checkpoint exactly once.** Training may rewrite the file at any moment, so hashing and loading must use the same bytes:
   - `blob = Path(ckpt).read_bytes()` and `sha = hashlib.sha256(blob).hexdigest()`.
   - `state = torch.load(io.BytesIO(blob), map_location="cpu")`.
   - Unwrap `state["state_dict"]`, or else `state["model"]`, when the key exists.
5. **Build the model and load weights.**
   - Construct `YoloRotBody(cfg.anchors_mask, 3, predict_angle=cfg.predict_angle, four_rot=cfg.four_rot, weighted_bbox=cfg.weighted_bbox, shared_cls=cfg.shared_cls, dual_bbox=True, shared_xy=cfg.shared_xy, shared_anchor=cfg.shared_anchor, raw_head=True, with_position=cfg.with_position, pe_channels=cfg.pe_channels, fpn_extra_upscaling=cfg.fpn_extra_upscaling, predict_strides=cfg.predict_strides, head_reduce_ratio=cfg.head_reduce_ratio, decoupled_head=cfg.decoupled_head, decoupled_bbox=cfg.decoupled_bbox, decoupled_cls=cfg.decoupled_cls, decoupled_obj=cfg.decoupled_obj, decoupled_obj_mode=cfg.decoupled_obj_mode)`.
   - Then `net.load_state_dict(state)` (strict) and `net = net.fuse().eval()`.
6. **Record the reference output.**
   - `x = torch.rand(1,3,640,640, generator=torch.Generator().manual_seed(0))`.
   - `ref = [t.clone() for t in net(x)]`, run under `torch.no_grad()`.
7. **Apply two graph-simplifying patches.** Both are numerically identical to the original model.
   - (a) **Bake the obj-head masks.** For each `k, m` in `list(getattr(net, "obj_head", {}).items())` where `m` is a `MaskedConv2d`:
     - create `conv = nn.Conv2d(m.in_channels, m.out_channels, 1, bias=m.bias is not None)`;
     - copy `m.weight * m.mask` into `conv.weight` and copy `m.bias` into `conv.bias`;
     - set `net.obj_head[k] = conv.eval()`;
     - print how many were baked (3 are expected).

     Why: OpenCV needs constant conv weights, not a runtime `Mul`.
   - (b) **Avoid zero-width tensors.** Set an instance attribute, `net._split_box_pred = split_box_pred`, where:
     ```python
     def split_box_pred(spec, box):
         pch = spec["prefix_ch"]
         assert spec["xy_per_cell"] == 0 and box.shape[1] - pch == spec["n_cells"] * spec["wh_per_cell"]
         return box[:, :pch], box[:, pch:]
     ```
     Why: the model's generic splitter slices a zero-width per-cell xy tensor (shape `1×10×0×H×W`) and concatenates it. That puts empty tensors in the graph.
8. **Check the patches changed nothing.**
   - Compute `pat = net(x)` under `torch.no_grad()`.
   - For each stride print `patched-vs-original stride <s>: max|diff|=<d>`.
   - Exit with `export patches changed the model output` if any `d > 1e-5`.
9. **Export** with the dynamo exporter and a static 1×3×640×640 input:
   ```python
   torch.onnx.export(net, (x,), f=out, input_names=["images"], output_names=["output_p5","output_p4","output_p3"],
                     opset_version=18, dynamo=True, external_data=False, dynamic_axes=None,
                     do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL, verbose=False)
   ```
   Leave the model's 4-rotation `torch.rot90` alone. The dynamo exporter turns it into `Transpose` plus a `Slice` with negative steps. OpenCV 5's new engine supports negative-step slices; OpenCV's classic engine does not, which is why Step 2 pins OpenCV 5.
10. **Post-export checks.** Each failure exits non-zero.
    - Load the file with `onnx.load(out)`.
    - Add metadata with `onnx.helper.set_model_props(m, {...})`:
      - `"yolo_rot.config"`: the config path.
      - `"yolo_rot.checkpoint"`: the checkpoint path with `/` separators.
      - `"yolo_rot.checkpoint_sha256"`: `sha`.
    - Save with `onnx.save(m, out)`, then run `onnx.checker.check_model(m)`.
    - Print `op types: <sorted unique op_type list>`. This is for diagnosis only.
    - Run onnxruntime: `ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})`.
      - The outputs must have shapes `(1,37,20,20)`, `(1,37,40,40)` and `(1,37,80,80)`, in that order.
      - For each output, `max|ort - pat| <= 1e-3`. Print it as `onnxruntime-vs-torch <name>: <d>`.
    - Final line: `exported <out> (<MB> MB) from <ckpt> sha256 <sha>`.

### Step 2 — OpenCV 5.0.0 and the CMake project
1. **Install OpenCV** (one time). OpenCV 5.0.0 is the official prebuilt Windows pack. Run in cmd:
   ```
   curl -L -o C:\tmp\opencv-5.0.0-windows.exe https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-windows.exe
   certutil -hashfile C:\tmp\opencv-5.0.0-windows.exe SHA256
   "C:\Program Files\7-Zip\7z.exe" x C:\tmp\opencv-5.0.0-windows.exe -oC:\Users\david\opencv-5.0.0 -y
   ```
   - The hash must be `9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63`.
   - Set `OpenCV_DIR = C:/Users/david/opencv-5.0.0/opencv/build`, the folder that contains `OpenCVConfig.cmake`.
2. **Create `cpp_dnn/CMakeLists.txt`:**
   - `cmake_minimum_required(VERSION 3.21)`, `project(yolo_rot_dnn LANGUAGES CXX)`, C++17 required.
   - `find_package(OpenCV 5 REQUIRED)`.
   - `add_executable(yolo_rot_dnn src/main.cpp src/detector.cpp)`, then `target_include_directories(... PRIVATE ${OpenCV_INCLUDE_DIRS})` and `target_link_libraries(... PRIVATE ${OpenCV_LIBS})`.
   - With MSVC: `/W4 /utf-8`.
   - On WIN32, copy the OpenCV runtime DLLs next to the exe. This includes the FFmpeg videoio plugin, which OpenCV loads at runtime and is therefore not a link dependency; video files will not open without it.
     - Use `_libdir = ${OpenCV_LIB_PATH}` when that variable is set; otherwise use `${OpenCV_DIR}`.
     - `get_filename_component(_bin "${_libdir}/../bin" ABSOLUTE)` and `file(GLOB _dlls "${_bin}/*.dll")`.
     - Drop the debug DLLs with `list(FILTER _dlls EXCLUDE REGEX "d\\.dll$")`.
     - Add `add_custom_command(TARGET yolo_rot_dnn POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls} $<TARGET_FILE_DIR:yolo_rot_dnn>)`.
     - If the list is empty, `message(WARNING "no OpenCV DLLs in ${_bin}; add it to PATH")`.
3. **Build commands** (cmd, from the repo root):
   ```
   call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   cmake -S cpp_dnn -B cpp_dnn\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/david/opencv-5.0.0/opencv/build
   cmake --build cpp_dnn\build
   ```
   The output is `cpp_dnn\build\yolo_rot_dnn.exe`. The build directory is already covered by the repo's `build/` gitignore rule.

### Step 3 — Model constants and the detector: `cpp_dnn/src/model_spec.hpp`, `detector.hpp`, `detector.cpp`
All code lives in `namespace yrd`.

**`model_spec.hpp`** (constants; must stay identical to the export script's `SPEC_*`):
- Sizes and decode parameters:
  - `kInputSize=640`
  - `kNumAnchors=5`
  - `kNumClasses=3`
  - `kRawChannels=4+6*5+3=37`
  - `kClsBase=34`
  - `kXyRadius=1.5f`
  - `kPadGray=128`
- `struct StrideSpec { int stride; std::array<std::array<float,2>,5> anchors; }` with anchors as (w,h) in pixels:
  - stride 32: `(142,110),(167,177),(192,243),(326,322),(459,401)`
  - stride 16: `(36,75),(56,65),(76,55),(74,101),(72,146)`
  - stride 8: `(12,16),(16,26),(19,36),(30,32),(40,28)`
- `kClassNames = {"person","vehicle","other"}`.
- `kClassColorsBgr = {{0,0,255},{255,0,0},{0,255,0}}`: red, blue, green.

**Raw ONNX output contract.** There are three tensors of shape `[1,37,H,W]`, with H=W=640/stride. They are matched to a `StrideSpec` by H, never by output order or name. Channels at each cell:
- 0,1: `tx0,ty0`, the logits for the box center. This is the only center that is used.
- 2,3: `tx45,ty45`. Unused.
- 4+3i, 5+3i, 6+3i for i=0..4: `tw0_i, th0_i, tobj0_i` (0° branch, anchor i).
- 19+3j, 20+3j, 21+3j for j=0..4: `tw45_j, th45_j, tobj45_j` (45° branch, anchor j).
- 34+c for c=0..2: class logits, shared by all 25 anchor pairs of the cell.

**`detector.hpp` API:**
- `struct Detection { int cls; float conf; float cx, cy, W0, H0, W45, H45, a, b, theta; };`
  - Pixel coordinates are in the 640×640 (rotated) canvas.
  - `W0,H0` and `W45,H45` are half-extents.
  - `a≥b` are the ellipse semi-axes.
  - `theta` is in radians, in [−π/2, π/2), in image coordinates (x right, y down).
- `enum class Engine { New, Classic };`
- `cv::Mat letterbox(const cv::Mat& bgr8uc3);`
- `cv::Mat rotateCanvas(const cv::Mat& canvas, int angleDeg);`
- `class YoloRotDetector`:
  - `YoloRotDetector(const std::string& onnxPath, Engine engine);` — throws `std::runtime_error`, or propagates `cv::Exception`.
  - `std::vector<Detection> detect(const cv::Mat& canvasBgr, float confThr, float nmsIou, double* inferMs);`
  - `const std::string& outputShapes() const;`

**`letterbox`** (reproduces the Python preprocessing):
- `scale = min(640.0/cols, 640.0/rows)` in double; `nw = int(cols*scale)`, `nh = int(rows*scale)`, truncated.
- The canvas is 640×640 `CV_8UC3`, filled with (128,128,128).
- The ROI is `Rect((640-nw)/2, (640-nh)/2, nw, nh)`.
- If `nw==cols && nh==rows`, copy the frame in unchanged; otherwise `cv::resize` into the ROI with `INTER_CUBIC`.

**`rotateCanvas`** rotates counter-clockwise by `a = ((angleDeg%360)+360)%360` degrees about `((W-1)/2, (H-1)/2)` = (319.5, 319.5):
- Return the input unchanged when `a==0`.
- `c,s`: use exact values for 90 (0,1), 180 (−1,0) and 270 (0,−1). Otherwise `cos/sin(a·π/180)`. Exact values keep the 90° steps as exact pixel permutations.
- Build `cv::Matx23d M(c, s, (1-c)*cx - s*cy, -s, c, s*cx + (1-c)*cy)` by hand. This is OpenCV's `getRotationMatrix2D` formula; that function is not declared in OpenCV 5's `imgproc.hpp`.
- `cv::warpAffine(canvas, out, M, canvas.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(128))`.

**Constructor:**
1. `net_ = cv::dnn::readNetFromONNX(path, engine==Engine::New ? cv::dnn::ENGINE_NEW : cv::dnn::ENGINE_CLASSIC)`. If `net_.empty()`, throw `failed to load ONNX model: <path>`.
2. `outNames_ = net_.getUnconnectedOutLayersNames()`.
3. Warm up by hand, not through `detect`, because `detect` assumes valid shapes. Build a 640×640 canvas filled with 128 gray, turn it into a blob exactly as `detect` does, call `setInput(blob, "images")`, then `forward(outs, outNames_)`.
4. Validate: there must be exactly 3 outputs. Each must have `dims==4`, `size[0]==1`, `size[1]==37`, `size[2]==size[3]`, and `640/size[2]` in {32,16,8}, with the three strides distinct.
5. Store `outputShapes()` as, for example, `[1,37,20,20] [1,37,40,40] [1,37,80,80]`.
6. On a validation failure, throw `unexpected model outputs <shapes>; expected three [1,37,H,W] tensors for strides 32/16/8 (raw-head ONNX from cpp_dnn/export_onnx.py)`. The stride lookup `specForGrid(H)` used by `detect` also throws this message for an unknown H.

**`detect`:**
1. Assert the canvas is 640×640 `CV_8UC3`.
2. `blob = cv::dnn::blobFromImage(canvas, 1.0/255.0, Size(), Scalar(), /*swapRB=*/true, /*crop=*/false, CV_32F)`. The model expects RGB in [0,1], NCHW.
3. `net_.setInput(blob, "images")`. Keep scale=1 and mean=0 on `setInput`; the new engine asserts on anything else.
4. Time only `net_.forward(outs, outNames_)` with `cv::getTickCount` and write the result to `*inferMs`.
5. For each output, clone it if it is not contiguous, then decode it with `decodeStride`.
6. Run class-agnostic NMS:
   - `boxes` are `Rect2d(cx-W0, cy-H0, 2W0, 2H0)`; `scores` are `conf`.
   - `cv::dnn::NMSBoxes(boxes, scores, 0.f, nmsIou, keep)`.
   - Return `cands[keep[k]]` in `keep` order, which is by descending confidence.

**`decodeStride`** per cell (gy, gx), where `s` = stride, `sig(v)=1/(1+exp(-v))`, and `c[k]` means channel k at this cell (`ptr[k*H*W + gy*W + gx]`):
1. `ci` = argmax over `c[34..36]`; strict `>` keeps the first maximum. `clsP = sig(c[34+ci])`. Skip the cell if `clsP < confThr`.
2. `cx = (3·sig(c[0]/1.5) − 1 + gx)·s` and `cy = (3·sig(c[1]/1.5) − 1 + gy)·s`. This is `2r·sig(t/r) − r + 0.5` with r=1.5.
3. For every i in 0..4 and j in 0..4:
   - `conf = sig(min(c[6+3i], c[21+3j])) · clsP`. The minimum is taken on the logits. Skip the pair if `conf < confThr`.
   - `bw0 = (2·sig(c[4+3i]))²·A[i].w` and `bh0 = (2·sig(c[5+3i]))²·A[i].h`, where A is this stride's anchors in pixels. Skip the pair if `bw0<2 || bh0<2`.
   - `bw45 = (2·sig(c[19+3j]))²·A[j].w` and `bh45 = (2·sig(c[20+3j]))²·A[j].h`.
   - Push `makeDetection(ci, conf, cx, cy, bw0/2, bh0/2, bw45/2, bh45/2)`.

**`makeDetection(int cls, float conf, float cx, float cy, float W0, float H0, float W45r, float H45r)`**, where `W45r,H45r` are the raw 45° half-extents. Compute in double and store as float:
1. `Sxx=W0²`, `Syy=H0²`, `Sxy=(W45r²−H45r²)/2`.
2. `tr=(Sxx+Syy)/2`; `r=sqrt(max(((Sxx−Syy)/2)²+Sxy², 0))`.
3. `a=sqrt(max(tr+r, 1e-12))`, `b=sqrt(max(tr−r, 0))`.
4. `th=0.5·atan2(2Sxy, Sxx−Syy)`, then `th += π/2; th −= π·floor(th/π); th −= π/2`.
5. The displayed 45° box is recomputed from the ellipse; the raw 45° extents are not used for it:
   - `sum=(a²+b²)/2`, `diff=(a²−b²)/2`, `s2=sin(2th)`.
   - `W45=sqrt(max(sum+diff·s2, 1e-12))`, `H45=sqrt(max(sum−diff·s2, 1e-12))`.
6. Store `W0,H0` as given.

### Step 4 — The app `cpp_dnn/src/main.cpp`: CLI, sources, rotation UI, drawing
**CLI.** Use `cv::CommandLineParser` with `--key=value` syntax and these keys:

| Key | Default | Notes |
|---|---|---|
| `help h usage ?` | — | |
| `model` | `<none>` | Required. The raw-head ONNX file. |
| `source` | `<none>` | Required. A digits-only string is a camera index; an extension in {.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp} (case-insensitive) is an image; anything else is passed to `VideoCapture` as a video file or URL. |
| `conf` | `0.5` | Must be in (0,1). |
| `nms` | `0.6` | Must be in (0,1]. |
| `angle` | `0` | Integer degrees counter-clockwise, normalized to [0,360). |
| `engine` | `new` | `new` or `classic`. |
| `max_frames` | `0` | Must be ≥0. The number of inference runs before exiting; 0 means no limit. |
| `headless` | — | Flag. |

- `parser.about(...)` lists the key bindings.
- `--help` prints the usage and exits with 0.
- On a parse or validation error, print `error: <msg>` plus the usage to stderr and exit with 1.

**Startup.**
- Build `YoloRotDetector`. On success print to stderr: `loaded <model> (engine <e>) in <ms> ms; outputs <shapes>`.
- Every error, whether `cv::Exception`, `std::exception`, or a source that fails to open, becomes `error: <what>` on stderr and exit code 1. The source-open messages are:
  - `cannot open camera <idx>`
  - `cannot open video source: <src>`
  - `cannot read image: <src>`

**Sources.**
- **Video:** `cv::VideoCapture cap(src)`, read sequentially in the main thread. End of stream prints `end of stream` to stderr and exits with 0.
- **Camera:** `cv::VideoCapture cap(idx)`, read through a `LatestFrameGrabber` class in main.cpp. Inference runs at roughly 1–3 FPS, and without this the camera buffer would serve stale frames.
  - The class holds `std::thread`, `std::mutex`, `cv::Mat frame_`, `long long seq_`, and `std::atomic<bool> stop_, failed_`. Declare `thread_` last.
  - The loop reads into a fresh local `Mat` each time, then stores it with `seq_++` under the lock. A read failure sets `failed_` and ends the loop.
  - `bool latest(Mat&, long long&)` returns a shallow copy.
  - The destructor sets `stop_` and joins.
  - Declare the grabber after `cap` so the grabber is destroyed first.
  - Whenever the main loop finds `failed_` set — before the first frame or mid-run, e.g. the camera was unplugged — print `error: camera stopped delivering frames` and exit with 1.
- **Image:** `cv::imread(src, IMREAD_COLOR)`; one frame.
- **Frame normalization:** convert 1-channel frames with GRAY2BGR and 4-channel frames with BGRA2BGR.

**Main loop.** State: `angle`, `paused`, `runs`, the current frame and its `frameId` (the 0-based index of the source frame; for a camera, the grabber sequence minus 1), plus `lastFrameId` and `lastAngle`.
1. **Get a frame.**
   - Image: load it once.
   - Video, when not paused (or when there is no frame yet): read the next frame.
   - Camera, when not paused: take the latest frame. If there is no new sequence number, call `waitKey(1)` in GUI mode (or sleep 5 ms when headless) and loop.
2. **Read the angle.** In GUI mode, `angle = getTrackbarPos("angle", win)`.
3. **Run inference** only when `frameId` or `angle` changed since the last run:
   - `canvas = letterbox(frame)`; `rot = rotateCanvas(canvas, angle)`; `dets = det.detect(rot, conf, nms, &ms)`; `++runs`.
   - Headless output, then `fflush(stdout)`:
     - `frame=<id> angle=<deg> infer_ms=<%.1f> dets=<n>`
     - one line per detection: `det cls=<name> conf=<%.4f> cx=<%.2f> cy=<%.2f> W0=<%.2f> H0=<%.2f> W45=<%.2f> H45=<%.2f> a=<%.2f> b=<%.2f> theta=<%.4f>`
   - GUI mode: `imshow(win, render(rot, dets, hud))`.
   - Stop when `max_frames>0 && runs>=max_frames`.
4. **Headless exit rules:** an image source stops after its one run. A video stops at end of stream or `max_frames`. A camera stops only at `max_frames` (or Ctrl+C).
5. **GUI keys.**
   - `k = waitKey(paused||image ? 30 : 1)`. Exit if `getWindowProperty(win, WND_PROP_VISIBLE) < 1`.
   - Bindings (on `k & 0xFF`):

     | Key | Action |
     |---|---|
     | `a` | +5 (counter-clockwise) |
     | `d` | −5 |
     | `A` | +45 |
     | `D` | −45 |
     | `r` | reset to 0 |
     | space | toggle pause (ignored for images) |
     | `q` or Esc | quit |

   - Angles are kept in [0,360). After any angle key, call `setTrackbarPos("angle", win, angle)`.
   - Window: `namedWindow("yolo_rot_dnn", WINDOW_AUTOSIZE)` plus `createTrackbar("angle", win, nullptr, 359)` initialized to `--angle`.
   - Inference is synchronous, so a trackbar change takes effect on the next run.
   - While paused or on an image, changing the angle re-runs inference on the same frame. This is how the user rotates a paused video frame.

**`render`** draws on a copy of the rotated canvas, then appends a 48-px black bar below it with `vconcat`, giving a 640×688 image. For each detection, using the class color (BGR) and `ink=(255−B,255−G,255−R)`:
1. **Ellipse:** 72 points, t_k = 2πk/72. `x = cx + a·cos t·cos th − b·sin t·sin th` and `y = cy + a·cos t·sin th + b·sin t·cos th`, each rounded with `cvRound`. Draw a closed `polylines` in black at thickness 4, then in the class color at thickness 2, both `LINE_AA`.
2. **0° box:** `rectangle(Point(int(cx−W0), int(cy−H0)), Point(int(cx+W0), int(cy+H0)), color, 1)`.
3. **45° box, dashed:**
   - The local corners (−W45,−H45), (W45,−H45), (W45,H45), (−W45,H45) map to `x=cx+(lx−ly)·√½` and `y=cy+(lx+ly)·√½`.
   - Draw each of the 4 edges as 10-px dashes and 6-px gaps, restarting the pattern on each edge, at thickness 1 with rounded endpoints.
4. **Labels,** drawn after all shapes:
   - Text `"<name> <conf %.2f>"`, `FONT_HERSHEY_SIMPLEX` 0.5, thickness 1; take `getTextSize` → `ts`, `base`.
   - Anchor at `x1=int(cx−W0)` and `y1=max(int(cy−H0), ts.height+base+4)`.
   - Filled rectangle from (x1, y1−ts.height−base−4) to (x1+ts.width+4, y1) in the class color.
   - Text at (x1+2, y1−base−2) in `ink`, `LINE_AA`.
5. **HUD,** white `FONT_HERSHEY_SIMPLEX` 0.5:
   - Line 1 at y=19: `angle %d deg CCW | infer %.0f ms | %.1f FPS | dets %zu`, with ` | PAUSED` appended while paused. FPS is an exponential moving average (α=0.2) of 1000 / (ms between consecutive inference runs).
   - Line 2 at y=40: `a/d +-5  A/D +-45  r reset  space pause  q/Esc quit`.

## Critical files & anchors
- `configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json` — the model being deployed. Input only.
- `logs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls/best_epoch_weights.pth` — the default weights. Training is live, so read the file once and record its sha256 in the ONNX metadata.
- `nets/yolo_rot.py` — `YoloRotBody`, `MaskedConv2d` and the instance method `_split_box_pred`, which the export overrides per instance. Imported only, never edited.
- `demo_rot_obb.py` — `YOLO_ROT`, the Python oracle for the parity check. Construct it with `(config, model_path, confidence, nms_iou, cuda=False)`. Its `_compute_detections(PIL.Image)` returns `[(cls_idx, conf, (cx, cy, W0, H0, W45, H45, a, b, theta)), ...]` in input-image pixels. Imported only.
- `cpp_dnn/src/model_spec.hpp` ↔ the `SPEC_*` constants in `cpp_dnn/export_onnx.py` — must stay identical. The export's spec guard enforces this.

## Verification
All commands run from `C:\Users\david\projects\yolov7-tiny-pytorch` with `C:\Python313\python.exe` and use `ONNX=model_data\rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls_rawhead.onnx`. Throwaway test files live in `C:\tmp\yolo_rot_dnn_verify\`, outside the repo.

1. **Export.** Run `C:\Python313\python.exe cpp_dnn\export_onnx.py`. Expect:
   - 3 masked convs baked;
   - every `patched-vs-original` value ≤1e-5;
   - every `onnxruntime-vs-torch` value ≤1e-3;
   - shapes 20/40/80 with 37 channels;
   - a final `exported ...` line; exit code 0.
2. **Build.** Run the Step 2.3 commands. `cpp_dnn\build\yolo_rot_dnn.exe` must exist next to `opencv_world*.dll` and `opencv_videoio_ffmpeg*.dll`. `yolo_rot_dnn.exe --help` prints the usage and exits with 0.
3. **Test data.** Write and run `C:\tmp\yolo_rot_dnn_verify\make_testdata.py`:
   - Letterbox `img/street.jpg` to 640×640 with PIL: `s=min(640/w,640/h)`, `int()` sizes, `BICUBIC`, gray (128,128,128) canvas, paste at `((640-nw)//2,(640-nh)//2)`. Save it as `street_640.png`.
   - Save `canvas.rotate(90)` (PIL: counter-clockwise, same size) as `street_640_rot90.png`.
   - Write `street.avi`: 20 MJPG frames at 1280×720 and 10 fps, using cv2 (4.13). Frame i is the street image resized with `INTER_AREA` and `np.roll`-ed 8·i px along x.
4. **Load and detect.** Run `yolo_rot_dnn.exe --model=%ONNX% --source=C:\tmp\yolo_rot_dnn_verify\street_640.png --headless`. Expect stderr `loaded ... outputs [1,37,20,20] [1,37,40,40] [1,37,80,80]`, stdout `frame=0 angle=0 ... dets=N` with N≥1 and N `det` lines, and exit code 0.
5. **Parity with the Python pipeline** (the new behavior). Write and run `C:\tmp\yolo_rot_dnn_verify\parity.py [conf=0.5]`. It adds the repo to `sys.path`, `chdir`s into it, and does the following:
   1. Read the ONNX `metadata_props`. Recompute the checkpoint's sha256; if it differs from `yolo_rot.checkpoint_sha256`, exit with `checkpoint changed since export; re-run export_onnx.py`.
   2. Build `YOLO_ROT(config=<cfg>, model_path=<checkpoint from metadata>, confidence=conf, nms_iou=0.6, cuda=False)`.
   3. Case **rot0:** Python runs on `street_640.png`; C++ runs on `street_640.png` with `--angle=0`.
   4. Case **rot90:** Python runs on `street_640_rot90.png`; C++ runs on `street_640.png` with `--angle=90`. Both inputs are exact pixel permutations, so this also checks the rotation direction and center convention.
   5. Run C++ with `--headless --conf=<conf>` and parse the `det` lines.
   6. For each Python detection with conf ≥ conf+0.02, the closest same-class C++ detection must meet all of these:
      - center distance ≤1.5 px;
      - |Δconf| ≤0.01;
      - W0, H0, W45, H45, a and b each within max(1 px, 1%);
      - when a−b > 0.05·a, the theta difference modulo π is ≤0.02 rad.
   7. Each C++ detection with conf ≥ conf+0.02 needs a same-class Python detection within 1.5 px.
   8. Case rot0 must compare at least 1 detection.
   9. Print `[rot0] ... PASS/FAIL` and `[rot90] ... PASS/FAIL`, and exit non-zero on any FAIL.
6. **Video path.** Run `yolo_rot_dnn.exe --model=%ONNX% --source=C:\tmp\yolo_rot_dnn_verify\street.avi --headless --max_frames=5 --angle=30`. Expect exactly 5 `frame=` lines with ids 0–4 and exit code 0. Report the `infer_ms` values as the CPU performance baseline.
7. **Error paths.** Each of these must print `error: ...` and exit with 1:
   - `--source=C:\tmp\nope.mp4 --headless`
   - `--model=C:\tmp\nope.onnx --source=...street_640.png --headless`
   - `--engine=bogus ...`
8. **USB camera.** Run `--source=0 --headless --max_frames=3`. Expect either 3 `frame=` lines or `error: cannot open camera 0` with exit code 1. Report which happened.
9. **GUI smoke test.** Run `--source=C:\tmp\yolo_rot_dnn_verify\street.avi --max_frames=15`. A window with the angle trackbar and the HUD bar opens and closes on its own with exit code 0.
10. **Manual checks (user),** on a real video and on the camera:
    - `a` rotates the view counter-clockwise; `d`, `A`, `D` and `r` behave as specified; the trackbar stays in sync.
    - Space pauses, and rotating while paused re-runs inference on the frozen frame.
    - Boxes and ellipses follow the objects at every angle.
    - `q`, Esc, and closing the window all exit.

## Assumptions & contingencies
**Assumptions**
- **Weights:** "the currently trained model" means the live 3-class run, using `best_epoch_weights.pth` by default, which is the convention the Python demos use. Pass `--weights last_epoch` or `--weights last_batch` to export newer weights. Re-export whenever training improves.
- **Hardware:** inference runs on the CPU only. The OpenCV 5 prebuilt pack and its new DNN engine have no GPU path. A CUDA build of OpenCV is not part of this plan.
- **Speed:** expect roughly 0.3–1 s per frame. [INFERENCE: estimated from about 60 GMAC per 640×640 frame for the 4-rotation model.]
- **What is displayed:** the rotated 640×640 network input, with detections in that frame. This matches the Python demo. The upright original frame is not shown.
- **Rotation settings:** counter-clockwise is positive, the rotation is bilinear, and the corners are filled with gray 128. The Python demo uses nearest-neighbor, so pixels differ slightly at non-90° angles.

**Contingencies**
- **C1 — OpenCV fails on the model** (`readNetFromONNX` or `forward` fails with `--engine=new`), **or parity fails while the export's onnxruntime check passed:**
  1. Add a `--flip-free` flag to the export script. Before exporting it does the following:
     - `net.register_buffer(f"_rev{n}", torch.arange(n-1, -1, -1), persistent=False)` for n in (640, 80, 40, 20);
     - temporarily replaces `torch.rot90` with a gather-based version:
       ```python
       def rot90_gather(t, k=1, dims=(0, 1)):
           d0, d1 = dims; k %= 4
           f = lambda x, d: x.index_select(d, getattr(net, f"_rev{x.shape[d]}"))
           return t if k == 0 else f(t, d1).transpose(d0, d1) if k == 1 else f(f(t, d0), d1) if k == 2 else f(t, d0).transpose(d0, d1)
       ```
     - restores `torch.rot90` afterwards;
     - keeps the Step 1.8 equivalence check;
     - after export, asserts that no `Slice` node has a negative constant step.
  2. Re-export, then retry with `--engine=new` and with `--engine=classic`. Make whichever engine passes parity the default; if both pass, keep `new`.
  3. If neither passes, stop and report the OpenCV error text or the parity diffs. Do not switch to a different inference framework.
- **C2 — the export call raises `TypeError` on `training=` or `do_constant_folding=`:** remove those two keyword arguments. Both are no-ops for the dynamo exporter.
- **C3 — the checkpoint can't be read while training writes it:** wait 60 s and rerun, or use `--weights last_epoch`.
- **C4 — the 7z extraction has a different layout:** find `OpenCVConfig.cmake` under `C:\Users\david\opencv-5.0.0` and use its folder as `OpenCV_DIR`.
- **C5 — parity finds no detection with conf ≥ 0.52 in case rot0:** rerun `parity.py 0.25` (both sides then use 0.25).
- **C6 — parity shows a systematic geometry error** (a constant center offset, or scaled sizes) **while confidences match:** recheck the decode against Step 3's channel map and formulas. The export checks already proved the ONNX file itself is correct.

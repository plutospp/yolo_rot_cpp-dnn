import argparse
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

try:
    import onnxruntime.quantization as ortq
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False
    CalibrationDataReader = object


class ImageCalibrationDataReader(CalibrationDataReader if HAS_ORT_QUANT else object):
    """Calibration data reader for static per-channel quantization (Static PCQ)."""

    def __init__(
        self,
        calib_dir=None,
        input_name="images",
        input_shape=(1, 3, 640, 640),
        num_samples=32,
    ):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.samples = []
        self.enum_data = None

        if calib_dir and Path(calib_dir).exists():
            image_paths = sorted(
                [
                    p
                    for p in Path(calib_dir).glob("*")
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                ]
            )[:num_samples]
            if image_paths:
                try:
                    from PIL import Image

                    for p in image_paths:
                        img = Image.open(p).convert("RGB")
                        # Letterbox resize to input_shape (640, 640)
                        w, h = img.size
                        target_h, target_w = input_shape[2], input_shape[3]
                        scale = min(target_w / w, target_h / h)
                        nw, nh = int(w * scale), int(h * scale)
                        img_resized = img.resize((nw, nh), Image.BICUBIC)

                        canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
                        canvas.paste(img_resized, ((target_w - nw) // 2, (target_h - nh) // 2))

                        arr = np.array(canvas, dtype=np.float32) / 255.0  # HWC, [0, 1]
                        arr = np.transpose(arr, (2, 0, 1))  # CHW
                        arr = np.expand_dims(arr, axis=0)  # NCHW: 1x3x640x640
                        self.samples.append(arr)
                except Exception as e:
                    print(f"Warning: failed to load calibration images: {e}", file=sys.stderr)

        # Fallback to deterministic gray / synthetic samples if no calibration images loaded
        if not self.samples:
            rng = np.random.RandomState(42)
            for i in range(num_samples):
                # Blend gray background with synthetic random shapes
                base = np.full(self.input_shape, 128.0 / 255.0, dtype=np.float32)
                noise = (rng.rand(*self.input_shape).astype(np.float32) - 0.5) * 0.2
                sample = np.clip(base + noise, 0.0, 1.0)
                self.samples.append(sample)

        self.rewind()

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.samples)
        try:
            return {self.input_name: next(self.enum_data)}
        except StopIteration:
            return None

    def rewind(self):
        self.enum_data = iter(self.samples)


def quantize_static_pcq(
    input_model_path,
    output_model_path,
    calibration_data_reader=None,
    per_channel=True,
    activation_type="uint8",
    weight_type="int8",
    quant_format="qdq",
    calib_dir=None,
    num_samples=32,
):
    """Performs Static Per-Channel Quantization (Static PCQ) on an ONNX model."""
    if not HAS_ORT_QUANT:
        raise RuntimeError("onnxruntime.quantization is required for quantization.")

    input_path = Path(input_model_path)
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if calibration_data_reader is None:
        calibration_data_reader = ImageCalibrationDataReader(
            calib_dir=calib_dir,
            num_samples=num_samples,
        )

    act_t = QuantType.QUInt8 if activation_type.lower() == "uint8" else QuantType.QInt8
    wt_t = QuantType.QInt8 if weight_type.lower() == "int8" else QuantType.QUInt8
    qfmt = QuantFormat.QDQ if quant_format.lower() == "qdq" else QuantFormat.QOperator

    print(
        f"Starting Static PCQ (Per-Channel Quantization):\n"
        f"  Input model:      {input_path}\n"
        f"  Output model:     {output_path}\n"
        f"  Per-channel:      {per_channel}\n"
        f"  Activation type:  {act_t}\n"
        f"  Weight type:      {wt_t}\n"
        f"  Quant format:     {qfmt}"
    )

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_data_reader,
        quant_format=qfmt,
        per_channel=per_channel,
        activation_type=act_t,
        weight_type=wt_t,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Successfully quantized model: {output_path} ({size_mb:.2f} MB)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static Per-Channel Quantization (Static PCQ) for ONNX models"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to input floating-point ONNX model file",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Path to output quantized ONNX model file (default: <input>_pcq.onnx)",
    )
    parser.add_argument(
        "--calib-dir",
        default="",
        help="Path to directory containing calibration images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of calibration samples (default: 32)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Enable per-channel weight quantization (default: True)",
    )
    parser.add_argument(
        "--quant-format",
        choices=["qdq", "qoperator"],
        default="qdq",
        help="Quantization format: qdq or qoperator (default: qdq)",
    )
    parser.add_argument(
        "--activation-type",
        choices=["uint8", "int8"],
        default="uint8",
        help="Activation quantization type (default: uint8)",
    )
    parser.add_argument(
        "--weight-type",
        choices=["int8", "uint8"],
        default="int8",
        help="Weight quantization type (default: int8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.model)
    if not input_path.exists():
        print(f"Error: model file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = input_path.parent / f"{input_path.stem}_pcq.onnx"

    quantize_static_pcq(
        input_model_path=input_path,
        output_model_path=out_path,
        per_channel=args.per_channel,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        quant_format=args.quant_format,
        calib_dir=args.calib_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

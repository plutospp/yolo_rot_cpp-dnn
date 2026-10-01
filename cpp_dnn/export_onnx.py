import argparse
import hashlib
import io
import os
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

try:
    from nets.yolo_rot import YoloRotBody, MaskedConv2d
    from utils.model_config import load_model_config
except ImportError:
    # Minimal fallback or dummy definitions when running in isolated test environments
    YoloRotBody = None
    MaskedConv2d = None
    load_model_config = None


SPEC_INPUT = [640, 640]
SPEC_STRIDES = [32, 16, 8]
SPEC_MASK = [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
SPEC_ANCHORS = [
    (12.0, 16.0), (16.0, 26.0), (19.0, 36.0), (30.0, 32.0), (40.0, 28.0),
    (36.0, 75.0), (56.0, 65.0), (76.0, 55.0), (74.0, 101.0), (72.0, 146.0),
    (142.0, 110.0), (167.0, 177.0), (192.0, 243.0), (326.0, 322.0), (459.0, 401.0)
]
SPEC_CLASSES = ["person", "vehicle", "other"]
SPEC_XY_RADIUS = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO-rot raw-head ONNX model")
    parser.add_argument("--config", default="configs/rot_dual_bbox_shared_xy_v2.5_decoupled_cls_obj_masked_3cls.json")
    parser.add_argument("--weights", choices=["best", "last_epoch", "last_batch"], default="best")
    parser.add_argument("--out", default="")
    parser.add_argument("--static-pcq", action="store_true", help="Perform static per-channel quantization (Static PCQ) after export")
    parser.add_argument("--calib-dir", default="", help="Directory containing calibration images for static PCQ")
    return parser.parse_args()


def check_spec_guard(cfg, config_path):
    def validate(cond, field, value, expected):
        if not cond:
            print(f"config {config_path}: {field}={value} does not match cpp_dnn/src/model_spec.hpp ({expected}); update both together", file=sys.stderr)
            sys.exit(1)

    validate(getattr(cfg, "head", None) == "dual_bbox", "head", getattr(cfg, "head", None), "dual_bbox")
    validate(bool(getattr(cfg, "shared_anchor", False)), "shared_anchor", getattr(cfg, "shared_anchor", False), True)
    validate(bool(getattr(cfg, "shared_xy", False)), "shared_xy", getattr(cfg, "shared_xy", False), True)
    validate(bool(getattr(cfg, "shared_cls", False)), "shared_cls", getattr(cfg, "shared_cls", False), True)
    validate(bool(getattr(cfg, "raw_head", False)), "raw_head", getattr(cfg, "raw_head", False), True)
    validate(getattr(cfg, "cls_loss_type", None) == "bce", "cls_loss_type", getattr(cfg, "cls_loss_type", None), "bce")

    validate(list(getattr(cfg, "input_shape", [])) == SPEC_INPUT, "input_shape", getattr(cfg, "input_shape", None), SPEC_INPUT)
    validate(list(getattr(cfg, "predict_strides", [])) == SPEC_STRIDES, "predict_strides", getattr(cfg, "predict_strides", None), SPEC_STRIDES)
    validate(list(getattr(cfg, "anchors_mask", [])) == SPEC_MASK, "anchors_mask", getattr(cfg, "anchors_mask", None), SPEC_MASK)

    # Parse anchors
    anchors_file = Path(cfg.anchors_path)
    if not anchors_file.exists():
        print(f"config {config_path}: anchors_path={cfg.anchors_path} does not exist", file=sys.stderr)
        sys.exit(1)
    line = anchors_file.read_text(encoding="utf-8").strip().splitlines()[0]
    vals = [float(x) for x in line.split(",") if x.strip()]
    parsed_anchors = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
    validate(parsed_anchors == SPEC_ANCHORS, "anchors", parsed_anchors, SPEC_ANCHORS)

    # Parse classes
    classes_file = Path(cfg.classes_path)
    if not classes_file.exists():
        print(f"config {config_path}: classes_path={cfg.classes_path} does not exist", file=sys.stderr)
        sys.exit(1)
    parsed_classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    validate(parsed_classes == SPEC_CLASSES, "classes", parsed_classes, SPEC_CLASSES)

    validate(getattr(cfg, "positive_grid_radius", 0.0) + 0.5 == SPEC_XY_RADIUS, "positive_grid_radius + 0.5", getattr(cfg, "positive_grid_radius", 0.0) + 0.5, SPEC_XY_RADIUS)


def split_box_pred_patch(spec, box):
    pch = spec["prefix_ch"]
    assert spec["xy_per_cell"] == 0 and box.shape[1] - pch == spec["n_cells"] * spec["wh_per_cell"]
    return box[:, :pch], box[:, pch:]


def main():
    args = parse_args()
    if load_model_config is None or YoloRotBody is None:
        print("Required model modules not found. Exiting.", file=sys.stderr)
        sys.exit(1)

    cfg = load_model_config(args.config)
    check_spec_guard(cfg, args.config)

    which = args.weights
    ckpt_path = Path(cfg.weights(which))
    if which == "best" and not ckpt_path.exists():
        ckpt_path = Path(cfg.weights("last_epoch"))

    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    blob = ckpt_path.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    state = torch.load(io.BytesIO(blob), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

    net = YoloRotBody(
        cfg.anchors_mask,
        3,
        predict_angle=cfg.predict_angle,
        four_rot=cfg.four_rot,
        weighted_bbox=cfg.weighted_bbox,
        shared_cls=cfg.shared_cls,
        dual_bbox=True,
        shared_xy=cfg.shared_xy,
        shared_anchor=cfg.shared_anchor,
        raw_head=True,
        with_position=cfg.with_position,
        pe_channels=cfg.pe_channels,
        fpn_extra_upscaling=cfg.fpn_extra_upscaling,
        predict_strides=cfg.predict_strides,
        head_reduce_ratio=cfg.head_reduce_ratio,
        decoupled_head=cfg.decoupled_head,
        decoupled_bbox=cfg.decoupled_bbox,
        decoupled_cls=cfg.decoupled_cls,
        decoupled_obj=cfg.decoupled_obj,
        decoupled_obj_mode=cfg.decoupled_obj_mode,
    )

    net.load_state_dict(state, strict=True)
    net = net.fuse().eval()

    x = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = [t.clone() for t in net(x)]

    # Bake obj-head masks
    baked = 0
    obj_head = getattr(net, "obj_head", {})
    items = list(obj_head.items()) if hasattr(obj_head, "items") else []
    for k, m in items:
        if isinstance(m, MaskedConv2d):
            conv = nn.Conv2d(m.in_channels, m.out_channels, 1, bias=m.bias is not None)
            conv.weight.data.copy_(m.weight.data * m.mask.data)
            if m.bias is not None:
                conv.bias.data.copy_(m.bias.data)
            obj_head[k] = conv.eval()
            baked += 1
    print(f"baked {baked} obj-head masked convs")

    # Split box pred patch
    net._split_box_pred = split_box_pred_patch

    with torch.no_grad():
        pat = net(x)

    for idx, s in enumerate(SPEC_STRIDES):
        diff = torch.max(torch.abs(pat[idx] - ref[idx])).item()
        print(f"patched-vs-original stride {s}: max|diff|={diff:.6e}")
        if diff > 1e-5:
            print("export patches changed the model output", file=sys.stderr)
            sys.exit(1)

    out_file = Path(args.out if args.out else f"model_data/{cfg.name}_rawhead.onnx")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            net,
            (x,),
            f=str(out_file),
            input_names=["images"],
            output_names=["output_p5", "output_p4", "output_p3"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            dynamic_axes=None,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
    except TypeError:
        torch.onnx.export(
            net,
            (x,),
            f=str(out_file),
            input_names=["images"],
            output_names=["output_p5", "output_p4", "output_p3"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            dynamic_axes=None,
            verbose=False,
        )

    m = onnx.load(str(out_file))
    onnx.helper.set_model_props(
        m,
        {
            "yolo_rot.config": str(args.config).replace("\\", "/"),
            "yolo_rot.checkpoint": str(ckpt_path).replace("\\", "/"),
            "yolo_rot.checkpoint_sha256": sha,
        },
    )
    onnx.save(m, str(out_file))
    onnx.checker.check_model(m)

    op_types = sorted(list(set(node.op_type for node in m.graph.node)))
    print(f"op types: {op_types}")

    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    ort_outs = session.run(None, {"images": x.numpy()})

    expected_shapes = [(1, 37, 20, 20), (1, 37, 40, 40), (1, 37, 80, 80)]
    for i, shape in enumerate(expected_shapes):
        if ort_outs[i].shape != shape:
            print(f"unexpected output shape for output {i}: {ort_outs[i].shape}, expected {shape}", file=sys.stderr)
            sys.exit(1)
        d = np.max(np.abs(ort_outs[i] - pat[i].detach().cpu().numpy()))
        print(f"onnxruntime-vs-torch output_p{5-i}: {d:.6e}")
        if d > 1e-3:
            print(f"onnxruntime output mismatch on output {i}: diff={d}", file=sys.stderr)
            sys.exit(1)

    mb = out_file.stat().st_size / (1024 * 1024)
    print(f"exported {out_file} ({mb:.2f} MB) from {ckpt_path} sha256 {sha}")

    if args.static_pcq:
        try:
            from quantize_onnx import quantize_static_pcq
            pcq_out = out_file.parent / f"{out_file.stem}_pcq.onnx"
            quantize_static_pcq(out_file, pcq_out, calib_dir=args.calib_dir)
        except Exception as e:
            print(f"Warning: static PCQ failed or skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

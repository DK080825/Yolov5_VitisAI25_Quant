import os, sys, torch
import torch.nn as nn
from pytorch_nndct.apis import Inspector

# Ưu tiên import từ repo hiện tại
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Kiểm tra import đúng file trong repo bạn
import models.common, models.yolo
print("[Import check] models.common ->", models.common.__file__)
print("[Import check] models.yolo   ->", models.yolo.__file__)

from models.yolo import Model

# ==== cấu hình ====
YAML   = "models/yolov5n.yaml"   # đổi theo model bạn dùng (v5s/l/x...)
WEIGHTS= "yolov5n.pt"             # .pt bạn muốn kiểm (COCO hay custom)
NC     = 80                       # số lớp
IMGSZ  = 640
TARGET = "DPUCZDX8G_ISA1_B4096"   # KV260
# ===================

def load_state_dict_from_pt(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        m = ckpt["model"]
        if hasattr(m, "state_dict"):
            return m.state_dict()
        if isinstance(m, dict):
            return m.get("state_dict", m)
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    return ckpt

def main():
    device = torch.device("cpu")  # Inspector chạy CPU là chắc nhất
    model = Model(YAML, ch=3, nc=NC).to(device).eval()

    sd = load_state_dict_from_pt(WEIGHTS)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")

    # Kiểm tra patch (ví dụ SiLU -> ReLU)
    has_silu = any(isinstance(m, nn.SiLU) for m in model.modules())
    print(f"[Check] Has SiLU? {has_silu}")

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
    inspector = Inspector(TARGET)
    inspector.inspect(model, (dummy,), device=device, output_dir="inspect_src", image_format="png")
    print("[OK] Inspector outputs -> inspect_src/")

if __name__ == "__main__":
    main()

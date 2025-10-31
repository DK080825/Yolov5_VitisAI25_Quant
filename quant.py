import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

from models.yolo import Model
from pytorch_nndct.apis import torch_quantizer

DIVIDER = '-'*50


class CustomImageDataset(Dataset):
    def __init__(self, label_dir, img_dir, width, height, transforms=None, image_exts=(".jpg",".jpeg",".png",".bmp")):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transforms = transforms
        self.height = height
        self.width = width

        self.items = []
        for filename in os.listdir(img_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() in image_exts:
                # chọn đúng ext thực tế của ảnh
                self.items.append((name, ext.lower()))
        self.items.sort()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, ext = self.items[idx]
        img_path = os.path.join(self.img_dir, name + ext)
        label_path = os.path.join(self.label_dir, name + ".txt")

        # ảnh RGB
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.Resize((self.height, self.width))(img)
        img = torchvision.transforms.ToTensor()(img)  # [0..1], (3,H,W)

        boxes_array, labels_array = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) < 5: 
                        continue
                    cls = int(vals[0])
                    cx, cy, w, h = map(float, vals[1:5])
                    x0 = (cx - w/2.0) * self.width
                    y0 = (cy - h/2.0) * self.height
                    x1 = (cx + w/2.0) * self.width
                    y1 = (cy + h/2.0) * self.height
                    labels_array.append(cls)
                    boxes_array.append([x0, y0, x1, y1])
        if len(boxes_array) == 0:
            boxes_array = [[0,0,1,1]]
            labels_array = [0]

        target = {
            "boxes": torch.as_tensor(boxes_array, dtype=torch.float32),
            "labels": torch.as_tensor(labels_array, dtype=torch.int64),
            "image_id": torch.tensor([idx + 1]),
        }
        

        return img, target


def load_state_dict_from_pt(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    
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


def build_model_from_yaml(yaml_path, nc, weights_pt):
    device = torch.device("cpu")  # PTQ CPU là chắc nhất
    model = Model(yaml_path, ch=3, nc=nc).to(device).eval()

    sd = load_state_dict_from_pt(weights_pt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")

    
    has_silu = any(isinstance(m, nn.SiLU) for m in model.modules())
    print(f"[Check] Model has SiLU: {has_silu}")
    return model


def quantize(build_dir, quant_mode, yaml_cfg, weights, dataset, nc=80, imgsz=640):
    out_dir = os.path.join(build_dir, 'quant_model')
    os.makedirs(out_dir, exist_ok=True)

    model = build_model_from_yaml(yaml_cfg, nc, weights)

    dummy = torch.randn(1, 3, imgsz, imgsz)
    quantizer = torch_quantizer(quant_mode, model, (dummy,), output_dir=out_dir)
    q_model = quantizer.quant_model.eval()

    img_dir = os.path.join(dataset, 'images')
    lbl_dir = os.path.join(dataset, 'labels')
    ds = CustomImageDataset(lbl_dir, img_dir, imgsz, imgsz)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        t0 = time.time()
        for i, (im, _) in enumerate(dl):
            _ = q_model(im)  
            if (i + 1) % 50 == 0:
                print(f"[calib] {i+1}/{len(dl)}")
        print(f"[calib done] {len(dl)} iters in {time.time()-t0:.2f}s")

    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"[OK] quant.json -> {out_dir}")
    elif quant_mode == 'test':
        try:
            quantizer.export_xmodel(deploy_check=False, output_dir=out_dir)
            print(f"[OK] .xmodel -> {out_dir}")
        except Exception as e:
            print("[Export] export_xmodel not supported in this VAI:", e)

    quantizer.shutdown()


def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b','--build_dir',  type=str, default='build', help='Build output dir')
    ap.add_argument('-q','--quant_mode', type=str, default='calib', choices=['calib','test'])
    ap.add_argument('-c','--yaml',       type=str, required=True,   help='Path to YOLO yaml (e.g. models/yolov5n.yaml)')
    ap.add_argument('-w','--weights',    type=str, required=True,   help='Path to .pt weights')
    ap.add_argument('-d','--dataset',    type=str, required=True,   help='Calib dir containing images/ and labels/')
    ap.add_argument('--nc',              type=int, default=80,      help='num classes')
    ap.add_argument('--imgsz',           type=int, default=640,     help='image size')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch:', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print('--build_dir :', args.build_dir)
    print('--quant_mode:', args.quant_mode)
    print('--yaml      :', args.yaml)
    print('--weights   :', args.weights)
    print('--dataset   :', args.dataset)
    print('--nc        :', args.nc)
    print('--imgsz     :', args.imgsz)
    print(DIVIDER)

    assert os.path.isdir(os.path.join(args.dataset, 'images')), "missing images/ in dataset"
    assert os.path.isdir(os.path.join(args.dataset, 'labels')), "missing labels/ in dataset"

    quantize(args.build_dir, args.quant_mode, args.yaml, args.weights, args.dataset, nc=args.nc, imgsz=args.imgsz)

if __name__ == '__main__':
    run_main()

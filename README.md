# YOLOv5 Vitis-AI Quantization and Deployment for KV260  
**Version:** Vitis-AI 2.5 | **Target:** DPUCZDX8G (KV260)

## Overview

This repository provides a **modified YOLOv5 pipeline** optimized for **AMD Xilinx Vitis-AI 2.5**,  
enabling conversion of trained PyTorch models (`.pt`) into deployable **INT8 `.xmodel`**  
files for the **KV260 DPU platform**.

The workflow covers:

-  Building a YOLOv5 model with Vitis-AI–compatible operators  
-  Performing **Post-Training Quantization (PTQ)** using PyTorch-NNDCT  
-  Compiling the quantized graph into a deployable `.xmodel` for **DPUCZDX8G (KV260)**

##  Environment Setup

Run inside the **Vitis-AI 2.5 Docker environment**:

```bash
docker run -it --rm \
  -v /path/to/your/project:/workspace/my_model \
  xilinx/vitis-ai-pytorch:2.5.0 \
  /bin/bash
```
## Base Repository

Start from the official **Ultralytics YOLOv5** repository (version 6.x – Python 3.7 compatible):

```bash
git clone -b v6.2 https://github.com/ultralytics/yolov5.git
cd yolov5
```
## Repository Modifications for Vitis-AI

The original Ultralytics YOLOv5 repository is written for standard PyTorch inference and exports.  
However, **Vitis-AI 2.5** has a **limited operator set**, so a few layers in YOLOv5 must be
modified before quantization and compilation. (https://docs.amd.com/r/2.5-English/ug1414-vitis-ai/Currently-Supported-Operators)

### 1. Replace SiLU with LeakyReLU
In `models/common.py` and `models/experimental.py` , replace:
```
self.act = nn.SiLU()
```
With:
```
self.act = nn.LeakyReLU(26/256, inplace=True)
```
After that, you need fine-tune the model for several epochs to recover accuracy before quantization.
```bash
python train.py --weights weights/best.pt --cfg models/yolov5n.yaml --data data/coco.yaml --epochs 50 --img 640
```
### 2. Post-Processing Refactor for Vitis-AI Compatibility
Since the `permute` and `view` operations in the YOLOv5 detection head are not supported by Vitis-AI,  
the final reshaping and decoding steps should be removed from the model’s forward function and  
reimplemented externally in the post-processing stage.

In `models/yolo.py`, remove the last layer in the detection head:
```
def forward(self, x):
        z = []  # inference output
        # print("anchors", self.anchors)
        # print("stride", self.stride)
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        return x
```
In `models/detect.py`, implement postprocessing function based on the original Detect class present in the ultralytics inside models/yolo.py.
```
def postprocessing(x, device, anchors, stride):
    """Post-process raw YOLOv5 output using anchors/stride FROM THE MODEL"""
    if isinstance(x, tuple):
        if isinstance(x[0], torch.Tensor) and x[0].dim() == 3:
            return x[0]
        x = x[0]

    # model already returned final preds
    if isinstance(x, torch.Tensor) and x.dim() == 3:
        return x

    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError(f"Unexpected output format from model: {type(x)} len={len(x) if hasattr(x,'__len__') else '-'}")

    nl = len(x)  # number of detection layers, usually 3
    grid = [torch.empty(0, device=device) for _ in range(nl)]
    anchor_grid = [torch.empty(0, device=device) for _ in range(nl)]
    z = []

    for i in range(nl):
        bs, ch, ny, nx = x[i].shape
        na = anchors.shape[1]   # 3
        no = ch // na           # 5 + nc

        # (bs, na, ny, nx, no)
        xi = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        # make grid if shape changed
        if grid[i].shape[2:4] != xi.shape[2:4]:
            grid[i], anchor_grid[i] = make_grid(nx, ny, i, anchors, stride, device)

        # decode like YOLOv5
        xy, wh, conf = xi.sigmoid().split((2, 2, no - 4), 4)
        xy = (xy * 2 + grid[i]) * stride[i]
        wh = (wh * 2) ** 2 * anchor_grid[i]

        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, -1, no))

    return torch.cat(z, 1)

def make_grid(nx, ny, i, anchors, stride, device):
    # anchors: (nl, na, 2) already /stride
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing='ij'
    )
    grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

    # bring anchors back to pixel space for this scale
    anchor_grid = (anchors[i].view(1, -1, 1, 1, 2) * stride[i])

    return grid, anchor_grid
```
## Quantizing with Vitis 2.5
### Step 1 — Calibration Mode (`quant_mode=calib`)

In **Calibration Mode**, the quantizer collects statistical information (e.g., activation min/max values)  
from representative images in order to determine optimal scaling factors for INT8 quantization.  
This step is essential to ensure numerical accuracy after quantization.

Run the calibration process:

```bash
  python ptq_yolov5_vai_quant.py \
  --yaml models/yolov5n.yaml \
  --weights refined_model.pt \
  --dataset calib_dir \
  --quant_mode calib \
  --build_dir build_y5n \
  --imgsz 640 \
  --nc 80
```
This exports a configuration file (quant_info.json) containing the collected quantization parameters
that will be reused in the next step.

### Step 2 — Test / Export Mode (quant_mode=test)

In Test Mode, the quantizer applies the previously collected calibration statistics
to generate a fully quantized model and export it in Vitis-AI–compatible .xmodel format.
This step can also be used to validate model performance before DPU compilation.

Run the export process:
```bash
  python ptq_yolov5_vai_quant.py \
  --yaml models/yolov5n.yaml \
  --weights refined_model.pt \
  --dataset calib_dir \
  --quant_mode test \
  --build_dir build_y5n \
  --imgsz 640 \
  --nc 80
```
This generates a deployable quantized model ready for compilation using the Vitis-AI compiler.
### Step 3 — Compilation with `vai_c_xir`

After generating the quantized `.xmodel`, the next step is to compile it using the  
**Vitis-AI XIR compiler (`vai_c_xir`)**. This process converts the quantized model into a
**DPU-specific binary** optimized for execution on the target FPGA (e.g., KV260).

Run the compilation command:

```bash
  vai_c_xir \
   -x build_y5n/quant_model/DetectionModel_int.xmodel \
   -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o kv260_xmodel
```

## Run inference

```bash
python detect.py --weights yolov5n.pt --yaml models/yolov5n.yaml --source data/images --data data/coco128.yaml --conf-thres 0.3  --conf-thres 0.35
```

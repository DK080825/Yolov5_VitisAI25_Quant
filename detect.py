# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, check_version)
from utils.torch_utils import select_device, smart_inference_mode


def load_custom_model(weights, yaml_cfg, data, device, half=False):
    """
    Load model from modified source code (models/yolo.py, common.py, experimental.py)
    
    Args:
        weights: path to .pt weights file
        yaml_cfg: path to model yaml config
        data: path to dataset yaml (for nc)
        device: torch device
        half: use FP16
    
    Returns:
        model, stride, names, pt
    """
    from models.yolo import Model
    import yaml
    
    # Load dataset config to get number of classes and class names
    with open(data, 'r') as f:
        data_dict = yaml.safe_load(f)
    nc = data_dict.get('nc', 80)
    
    # Extract class names from data yaml
    names = data_dict.get('names', None)
    if names is None:
        names = [f'class{i}' for i in range(nc)]
    elif isinstance(names, dict):
        # Convert dict to list (e.g., {0: 'person', 1: 'car'})
        names = [names[i] for i in range(nc)]
    
    # Build model from yaml config with modified source
    LOGGER.info(f'Loading model from modified source: {yaml_cfg}')
    model = Model(yaml_cfg, ch=3, nc=nc).to(device)
    
    # Load weights
    weights_path = Path(weights[0]) if isinstance(weights, list) else Path(weights)
    LOGGER.info(f'Loading weights: {weights_path}')
    ckpt = torch.load(weights_path, map_location=device)
    
    # Try to extract names from checkpoint
    if isinstance(ckpt, dict):
        # Check if names are stored in checkpoint
        if 'names' in ckpt:
            ckpt_names = ckpt['names']
            if isinstance(ckpt_names, dict):
                names = [ckpt_names[i] for i in sorted(ckpt_names.keys())]
                LOGGER.info('Using class names from checkpoint (converted from dict)')
            elif isinstance(ckpt_names, list):
                names = ckpt_names
                LOGGER.info('Using class names from checkpoint')
        
        # Extract state_dict
        if 'model' in ckpt:
            model_ckpt = ckpt['model']
            # Check if model object has names attribute
            if hasattr(model_ckpt, 'names') and model_ckpt.names:
                ckpt_names = model_ckpt.names
                if isinstance(ckpt_names, dict):
                    names = [ckpt_names[i] for i in sorted(ckpt_names.keys())]
                    LOGGER.info('Using class names from model checkpoint (converted from dict)')
                elif isinstance(ckpt_names, list):
                    names = ckpt_names
                    LOGGER.info('Using class names from model checkpoint')
            state_dict = model_ckpt.state_dict() if hasattr(model_ckpt, 'state_dict') else model_ckpt
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        # Check if checkpoint itself is a model with names
        if hasattr(ckpt, 'names') and ckpt.names:
            ckpt_names = ckpt.names
            if isinstance(ckpt_names, dict):
                names = [ckpt_names[i] for i in sorted(ckpt_names.keys())]
                LOGGER.info('Using class names from checkpoint (converted from dict)')
            elif isinstance(ckpt_names, list):
                names = ckpt_names
                LOGGER.info('Using class names from checkpoint')
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt
    
    # Final safety check: ensure names is always a list
    if not isinstance(names, list):
        if isinstance(names, dict):
            try:
                names = [names[i] for i in sorted(names.keys())]
                LOGGER.info(f'Converted names from dict to list ({len(names)} classes)')
            except Exception as e:
                LOGGER.warning(f'Failed to convert names dict: {e}')
                names = [f'class{i}' for i in range(nc)]
        else:
            LOGGER.warning(f'Names is unexpected type: {type(names)}, creating default names')
            names = [f'class{i}' for i in range(nc)]
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(f'Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}')
    
    # Set model to eval mode
    model.eval()
    
    # Apply FP16 if needed
    if half:
        model.half()
    
    # Extract model attributes
    stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
    
    # Assign names to model (ensure it's a list)
    model.names = names
    pt = True  # PyTorch model flag
    
    LOGGER.info(f'Model loaded: stride={stride}, classes={len(names)}')
    
    # Safe display of class names
    try:
        if isinstance(names, list) and len(names) > 0:
            display_names = names[:5] if len(names) > 5 else names
            LOGGER.info(f'Class names: {display_names}{"..." if len(names) > 5 else ""}')
        else:
            LOGGER.info(f'Class names: {names}')
    except Exception as e:
        LOGGER.warning(f'Could not display class names: {e}')
    
    return model, stride, names, pt


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        yaml='',  # model yaml config (use modified source if provided)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    if yaml:  # Use modified source code
        model, stride, names, pt = load_custom_model(weights, yaml, data, device, half)
        use_custom_postprocess = True
        detect_layer = model.model[-1]                  # last layer is Detect()
        model_anchors = detect_layer.anchors.to(device) # shape (3,3,2), already /stride
        model_stride  = detect_layer.stride.to(device)  # tensor([8.,16.,32.])
    else:  # Use original DetectMultiBackend
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        use_custom_postprocess = False
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz)) if hasattr(model, 'warmup') else None  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            
            if use_custom_postprocess:
                # Custom model returns raw output, need postprocessing
                predi = model(im)
                pred = postprocessing(predi, device, model_anchors, model_stride)
                
                # Verify output format for NMS
                if pred.dim() != 3 or pred.shape[-1] < 5:
                    raise ValueError(f"Invalid prediction shape for NMS: {pred.shape}. Expected [bs, num_pred, 4+1+nc]")
            else:
                # DetectMultiBackend returns processed output
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    
                    # Get class name - handle both list and dict formats
                    if isinstance(names, dict):
                        class_name = names.get(c, f'class{c}')
                    elif isinstance(names, list) and c < len(names):
                        class_name = names[c]
                    else:
                        class_name = f'class{c}'
                    
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, class_name, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        label = None if hide_labels else (class_name if hide_conf else f'{class_name} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / class_name / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5n.pt', help='model path or triton URL')
    parser.add_argument('--yaml', type=str, default='', help='model yaml config (use modified source if provided)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
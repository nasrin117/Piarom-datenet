import torch
import torch.backends.cudnn as cudnn

from ultralytics import YOLO
from argparse import ArgumentParser

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data", type=str, default="db/new classify data")
parser.add_argument("--models", type=list, default=[
    "models/yolo-cls/yolov8n-cls.pt",
    "models/yolo-cls/yolov8s-cls.pt",
    "models/yolo-cls/yolov8m-cls.pt"])
parser.add_argument("--augments", type=list, default=['custom', 'default', False])
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--imgsz", type=int, default=480)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--export", type=bool, default=True)
parser.add_argument("--format", type=str, default="onnx")
args = parser.parse_args()

CLASSIFY_MODEL_PATHS=args.models
AUGMENTS=args.augments
CLASSIFY_DB_PATH=args.data

for model_path in CLASSIFY_MODEL_PATHS:
    for augment in AUGMENTS:
        device = 0 if torch.cuda.is_available() and args.device == "cuda" else "cpu"

        log = f"| Model:{model_path.split('/')[-1]}, Freeze:{args.freeze}, Batch:{args.batch}, Epochs:{args.epochs}, Augment:{augment} |"
        print("-"*len(log)) 
        print(log)
        print("-"*len(log))

        model = YOLO(model=model_path, task="classify")

        want_to_freeze = [f'model.{x}.' for x in range(args.freeze)]
        freezed_layers = []
        for k, v in model.named_parameters():
            if any(x in k for x in want_to_freeze):
                v.requires_grad = False
                freezed_layers.append(k)
                
        if augment == 'custom':
            model.train(
                data=CLASSIFY_DB_PATH, 
                device=device,
                batch=args.batch,
                imgsz=args.imgsz, 
                epochs=args.epochs, 
                freeze=freezed_layers, 
                bgr=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                mixup=1.0,
                scale=0.1,
                fliplr=0.0,
                flipud=0.0,
                mosaic=1.0,
                erasing=0.0,
                shear=-15.0,
                degrees=15.0,
                translate=0.1,
                copy_paste=1.0,
                auto_augment=augment,
                perspective=0.0005,
                )
            
        elif augment == 'default':
            model.train(
                data=CLASSIFY_DB_PATH, 
                device=device,
                batch=args.batch,
                imgsz=args.imgsz, 
                epochs=args.epochs, 
                freeze=freezed_layers, 
                )
            
        elif augment == False:
            model.train(
                data=CLASSIFY_DB_PATH, 
                device=device,
                batch=args.batch,
                imgsz=args.imgsz, 
                epochs=args.epochs, 
                freeze=freezed_layers, 
                bgr=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                mixup=0.0,
                scale=0.0,
                shear=0.0,
                fliplr=0.0,
                flipud=0.0,
                mosaic=0.0,
                erasing=0.0,
                degrees=0.0,
                translate=0.0,
                copy_paste=0.0,
                perspective=0.0,
                auto_augment=augment,
                )

        if args.export:
            model.export(format=args.format, dynamic=True)
            print(f"Model exported to {args.format} format")

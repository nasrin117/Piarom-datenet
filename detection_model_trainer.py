import torch
from ultralytics import YOLO
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

parser = ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data", type=str, default="db/detection data/data.yaml")
parser.add_argument("--model", type=str, default="models/yolo-detect/yolov10m.pt")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=1280)
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--export", type=bool, default=True)
parser.add_argument("--format", type=str, default="onnx")
args = parser.parse_args()

DETECTION_MODEL_PATH=args.model
DETECTION_DB_PATH=args.data

device = 0 if torch.cuda.is_available() and args.device == "cuda" else "cpu"

log = f"| Model:{DETECTION_MODEL_PATH.split('/')[-1]}, Freeze:{args.freeze}, Batch:{args.batch}, Epochs:{args.epochs} |"
print("-"*len(log)) 
print(log)
print("-"*len(log))

yolo = YOLO(model=DETECTION_MODEL_PATH, task="detect")
want_to_freeze = [f'model.{x}.' for x in range(args.freeze)]
freezed_layers = []
for k, v in yolo.named_parameters():
    if any(x in k for x in want_to_freeze):
        v.requires_grad = False
        freezed_layers.append(k)

yolo.train(
    data=DETECTION_DB_PATH, 
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch, 
    freeze=args.freeze, 
    device=device,
    save_txt=True,
    save_conf=True,
    )

if args.export:
    yolo.export(format=args.format, dynamic=True)
    print(f"Exported to {args.format} format")

import onnxruntime
from argparse import ArgumentParser
from services.modules.app_modules import AppFunctions, app

parser = ArgumentParser()
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--cls_model', type=str, default="runs/classify/train/weights/best.onnx")
parser.add_argument('--detect_model', type=str, default="runs/detect/train/weights/best.onnx")
parser.add_argument('--cls_imgsz', type=int, default=480)
parser.add_argument('--detect_imgsz', type=int, default=1280)
args = parser.parse_args()

CLASSIFY_MODEL_PATH = args.cls_model
DETECTION_MODEL_PATH = args.detect_model
CLASSIFY_IMGSZ = args.cls_imgsz
DETECTION_IMGW = args.detect_imgsz

classify_model = onnxruntime.InferenceSession(CLASSIFY_MODEL_PATH)
detection_model = onnxruntime.InferenceSession(DETECTION_MODEL_PATH)


app_functions = AppFunctions(classify_model, detection_model, CLASSIFY_IMGSZ, DETECTION_IMGW)

if __name__ == "__main__":
    app(app_functions, args)

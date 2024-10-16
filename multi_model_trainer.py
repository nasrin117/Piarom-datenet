import torch
from torchvision import models
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from classify_model_trainer_pytorch import model_trainer
from services.modules.lightning_modules import create_dataloaders

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="db/classification data")
    parser.add_argument("--models", type=list, default=[
        'mobilenet_v3_large',
        'mobilenet_v3_small',
        'mobilenet_v2',
        'resnet18'])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--export", type=bool, default=True)
    parser.add_argument("--format", type=str, default="onnx")
    args = parser.parse_args()

    num_classes =args.num_classes
    learning_rate = args.learning_rate
    CLASSIFY_DB_PATH_PYTORCH = args.data
    train_dir = f"{CLASSIFY_DB_PATH_PYTORCH}/train"
    val_dir = f"{CLASSIFY_DB_PATH_PYTORCH}/val"

    model_architectures = args.models
    
    train_loader, val_loader = create_dataloaders(args, train_dir, val_dir)

    for model in model_architectures:
        try:
            model_architecture = getattr(models, model)
        except AttributeError:
            raise ValueError(f"Model {args.model} not found in torchvision.models")
        
        log = f"{model_architecture.__name__}/Augment: {args.augment}/Imgsz: {args.imgsz}"
        print(f"+{'-'*(len(log)+2)}+\n| {log} |\n+{'-'*(len(log)+2)}+")
        logger = TensorBoardLogger(save_dir=f"runs/pytorch/", name=log)

        model_trainer(args, model_architecture, num_classes, learning_rate, logger, train_loader, val_loader)

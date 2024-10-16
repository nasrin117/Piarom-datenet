import torch
from torchvision import models
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from services.modules.lightning_modules import LightningModel, create_dataloaders, show_samples

def model_trainer(args, model_architecture, num_classes, learning_rate, logger, train_loader, val_loader):

    model = LightningModel(model_architecture=model_architecture, num_classes=num_classes, learning_rate=learning_rate)

    checkpoint_callback = ModelCheckpoint(filename="best_model", monitor="val/accuracy", mode="max", save_top_k=1)
    trainer = Trainer(accelerator=args.device, max_epochs=args.epochs, logger=logger, callbacks=checkpoint_callback)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    show_samples(train_loader, args.device, logger, args.batch)

    if args.export:
        best_model_path = checkpoint_callback.best_model_path
        best_model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path, 
            weights_only=True, 
            model_architecture=model_architecture,
            num_classes=num_classes,
            learning_rate=learning_rate,
            )
        
        input_sample = torch.randn((1, 3, args.imgsz, args.imgsz))
        best_model.eval()
        best_model.to_onnx(
            best_model_path.replace("ckpt", args.format), 
            input_sample, 
            export_params=True,
            )
        
        print(f"Model exported to {args.format} format")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="db/classification data")
    parser.add_argument("--model", type=str, default="mobilenet_v2")
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

    train_dir = f"{args.data}/train"
    val_dir = f"{args.data}/val"

    try:
        model_architecture = getattr(models, args.model)
    except AttributeError:
        raise ValueError(f"Model {args.model} not found in torchvision.models")

    train_loader, val_loader = create_dataloaders(args, train_dir, val_dir)

    log = f"{model_architecture.__name__}/Augment: {args.augment}/Imgsz: {args.imgsz}"
    print(f"+{'-'*(len(log)+2)}+\n| {log} |\n+{'-'*(len(log)+2)}+")
    logger = TensorBoardLogger(save_dir=f"runs/pytorch/", name=log)

    model_trainer(args, model_architecture, args.num_classes, args.learning_rate, logger, train_loader, val_loader)


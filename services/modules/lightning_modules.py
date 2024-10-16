import glob
import torch
import torch.nn as nn
import subprocess as sp
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as T

from PIL import Image, ImageOps
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Recall, F1Score, Accuracy, Precision


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.replace_dict = {
            "disintegrated": 0,
            "first_class_black": 1,
            "first_class_gold": 2,
            "low_skin_separated_black": 3,
            "low_skin_separated_gold": 4,
            "mashed": 5,
            "moldy": 6,
            "nightingale_eaten": 7,
            "skin_separated_black": 8,
            "skin_separated_gold": 9,
            "streaky": 10,
        }
        self.data = []
        self.labels = []
        for image_path in sorted(glob.glob(f"{data_dir}/**/*.jpg", recursive=True)):
            self.data.append(image_path)
            self.labels.append(self.convert_classes(image_path.split("/")[-1].split("_2024")[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        # max_size = max(image.size)
        # image = self.resize_with_padding(image, (max_size, max_size))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def convert_classes(self, label):
        for key, value in self.replace_dict.items():
            label = label.replace(key, str(value))
        return int(label)

    def resize_with_padding(self, img, expected_size): 
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding, (255, 255, 255))


class LightningModel(pl.LightningModule):
    def __init__(self, model_architecture, num_classes=11, learning_rate=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.model = model_architecture(num_classes=self.num_classes)

        self.recall = Recall(num_classes=self.num_classes, task='multiclass')
        self.f1score = F1Score(num_classes=self.num_classes, task='multiclass')
        self.accuracy = Accuracy(num_classes=self.num_classes, task='multiclass')
        self.precision = Precision(num_classes=self.num_classes, task='multiclass')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        f1score = self.f1score(outputs, labels)
        recall = self.recall(outputs, labels)
        precision = self.precision(outputs, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True)
        self.log("train/f1score", f1score, on_step=True, on_epoch=True)
        self.log("train/recall", recall, on_step=True, on_epoch=True)
        self.log("train/precision", precision, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        f1score = self.f1score(outputs, labels)
        recall = self.recall(outputs, labels)
        precision = self.precision(outputs, labels)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/accuracy", accuracy, on_step=True, on_epoch=True)
        self.log("val/f1score", f1score, on_step=True, on_epoch=True)
        self.log("val/recall", recall, on_step=True, on_epoch=True)
        self.log("val/precision", precision, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        gpu_usage = self.get_gpu_memory()
        loss = self.criterion(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        f1score = self.f1score(outputs, labels)
        recall = self.recall(outputs, labels)
        precision = self.precision(outputs, labels)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        self.log("test/accuracy", accuracy, on_step=True, on_epoch=True)
        self.log("test/f1score", f1score, on_step=True, on_epoch=True)
        self.log("test/recall", recall, on_step=True, on_epoch=True)
        self.log("test/precision", precision, on_step=True, on_epoch=True)
        self.log("test/gpu_usage", gpu_usage, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }}

    def get_gpu_memory(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
        return float(memory_use_info[0][:-4])
    
    
def create_dataloaders(args, train_dir, val_dir):
    train_transform = T.Compose([
        T.Resize((args.imgsz, args.imgsz)), 
        T.RandomApply([T.RandomRotation(degrees=15)], p=0.5), 
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0))], p=0.5), 
        T.RandomPerspective(distortion_scale=0.1, p=0.5), 
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255), 
        T.ToTensor(), 
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_transform = T.Compose([
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_dataset = CustomDataset(train_dir, train_transform)
    val_dataset = CustomDataset(val_dir, val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, num_workers=16,shuffle=False)

    return train_loader, val_loader


def show_samples(train_loader, device, writer, batch_size):
    for i, batch in enumerate(train_loader):
        if i == 3:
            break
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        grid = vutils.make_grid(inputs, nrow=batch_size, normalize=True, value_range=(-1, 1))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"{writer.log_dir}/samples_{i}.png")

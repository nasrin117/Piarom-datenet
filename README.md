# Piarom Date Defect Detection

## Overview

Grading and quality control of Piarom dates—a premium and high-value variety cultivated predominantly in
Iran—present significant challenges due to the complexity and variability of defects, as well as the absence of
specialized automated systems tailored to this fruit. Traditional manual inspection methods are labor-intensive,
time-consuming, and prone to human error, while existing AI-based sorting solutions are insufficient for
addressing the nuanced characteristics of Piarom dates. In this study, we propose an innovative deep learning
framework designed specifically for the real-time detection, classification, and grading of Piarom dates.
Leveraging a custom dataset comprising over 9,900 high-resolution images annotated across 11 distinct defect
categories, our framework integrates state-of-the-art object detection algorithms and Convolutional Neural
Networks (CNNs) to achieve high precision in defect identification. Furthermore, we employ advanced
segmentation techniques to estimate the area and weight of each date, thereby optimizing the grading process
according to industry standards. Experimental results demonstrate that our system significantly outperforms
existing methods in terms of accuracy and computational efficiency, making it highly suitable for industrial
applications requiring real-time processing. This work not only provides a robust and scalable solution for
automating quality control in the Piarom date industry but also contributes to the broader field of AI-driven food
inspection technologies, with potential applications across various agricultural products.

## Datasets

### Primary Data

To generate the primary dataset:

- A black background was set as the bottom layer with a white layer on top, allowing easy separation of the dates from the background by color contrast.
- In each image, 50 dates were placed in a 5x10 grid (5 columns, 10 rows), resulting in 900 samples per class.
- White background dimensions: **32 cm x 45 cm**.

### Cropping Box Method

Images were cropped to remove the black borders and preserve the true physical ratio of the dates, ensuring accurate data for both object detection and classification tasks.

## Object Detection Workflow

After cropping the primary images, we utilized **Roboflow** to label and draw bounding boxes around the "Dates" in each image. Data augmentation was applied to increase the sample size and variety, ensuring robust model training.

### Train-Test Split

- **Train:** 80%
- **Test:** 20%

### Augmentation Techniques

- **Flip:** Horizontal and Vertical
- **Rotation:** 90° (Upside Down)
- **Saturation:** ±15%
- **Brightness:** ±15%
- **Exposure:** ±6%
- **Blur:** Up to 0.6px
- **Noise:** Up to 0.5% of pixels
- Additional augmentations were applied using YOLO-based techniques.

## Classification Workflow

For classification, individual date images were cropped and labeled according to their respective defect or quality category.

### Train & Test Split

- **Train:** ~78%
- **Test:** ~22%

### Categories

The dataset includes 11 distinct categories:

1. Disintegrated
2. Mashed
3. Moldy
4. Nightingale Eaten
5. Streaky
6. First Class Golden
7. Low Skin Separated Golden
8. Skin Separated Golden
9. First Class Black
10. Low Skin Separated Black
11. Skin Separated Black

These categories represent various defects, quality grades, and color differences in the Piarom dates.

## Data Augmentation

Augmentations were applied using the **Albumentations** library, including the following filters:

- **Flip:** Horizontal, Vertical
- **Rotate:** 90°
- **Saturation, Brightness, Exposure:** Small variations were applied to simulate real-world lighting conditions.
- **Blur and Noise:** Introduced to make the model more robust to image quality issues.

## Dataset Construction

This dataset was created in a controlled environment to ensure consistency and accuracy during both training and evaluation. Each image was captured under the same conditions, with a black and white background setup, ensuring precise segmentation and accurate model training.

## Data Input & Output

- **Bounding Box Sorting Method:** Bounding boxes were sorted by their y-coordinate (descending order) and then by their x-coordinate (descending order) for better object localization in the images. The method ensured that boxes were sorted according to their positions in the image, not just by coordinates.

## Final Dataset

- **Total Images:** 9900 labeled images
- **Per Category:** 900 images per defect or quality class

This dataset is ready for training and evaluation of deep learning models for object detection and classification, aiming to automate the sorting and grading of Piarom dates based on defects, quality, and color.

## Links

- [**Google Drive Link For `Dataset`**](https://drive.google.com/drive/folders/1-YQTPSHah-aBXTYcO92mManBZ1bvMLRs?usp=sharing)
- [**Google Drive Link For `weights`**](https://drive.google.com/drive/folders/18NVHJ_KA9mcS9ZJmWursoeHI9XXM27NS?usp=sharing)

## Requirements

To run the files, the following packages are required:

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pypi.org/project/torchvision/)
- [Albumentations](https://albumentations.ai/)
- [PyTorch Lightning](https://lightning.ai/)
- [YOLO Framework](https://docs.ultralytics.com/)

you can run this command:

```shell
pip3 install -r requirements.txt
```

## Running the Files

1. **Training the Classification Model:**
   To run the classification model trainer:

   ```bash
   python classify_model_trainer_pytorch.py # for training a single model
      --device # type=str, default="cuda"
      --data # type=str, default="db/classification data"
      --model # type=str, default='mobilenet_v2'
      --epochs # type=int, default=50
      --num_classes # type=int, default=11
      --learning_rate # type=float, default=0.001
      --imgsz # type=int, default=480
      --batch # type=int, default=1
      --freeze # type=int, default=0
      --augment # type=bool, default=True
      --export # type=bool, default=True
      --format # type=str, default="onnx" 

   python multi_model_trainer.py # for training multi model
      --device # type=str, default="cuda"
      --data # type=str, default="db/classification data"
      --models # type=list, default=['mobilenet_v3_large','mobilenet_v3_small','mobilenet_v2','resnet18']
      --epochs # type=int, default=50
      --num_classes # type=int, default=11
      --learning_rate # type=float, default=0.001
      --imgsz # type=int, default=480
      --batch # type=int, default=1
      --freeze # type=int, default=0
      --augment # type=bool, default=True
      --export # type=bool, default=True
      --format # type=str, default="onnx"
   ```

    **or run this code for yolo model:**

   ```bash
   python classify_model_trainer_yolo.py
      --device # type=str, default="cuda"
      --data # type=str, default="db/new classify data"
      --models # type=list, default=["models/yolo-cls/yolov8n-cls.pt", "models/yolo-cls/yolov8s-cls.pt", "models/yolo-cls/yolov8m-cls.pt"]
      --augments # type=list, default=['custom', 'default', False]
      --epochs # type=int, default=150
      --imgsz # type=int, default=480
      --batch # type=int, default=8
      --freeze # type=int, default=0
      --export # type=bool, default=True
      --format # type=str, default="onnx"
   ```

2. **Training the Detection Model:**
   To run the classification model trainer:

   ```bash
   python detection_model_trainer.py
      --device # type=str, default="cuda"
      --data # type=str, default="db/detection data/data.yaml"
      --model # type=str, default="models/yolo-detect/yolov10m.pt"
      --epochs # type=int, default=50
      --imgsz # type=int, default=1280
      --batch # type=int, default=2
      --freeze # type=int, default=0
      --export # type=bool, default=True
      --format # type=str, default="onnx"
   ```

3. **Infernce the models:**
   To get the inference of models:

   ```bash
   python inference_app.py
   --port # type=int, default=8000
   --cls_model # type=str, default="runs/classify/train/weights/best.onnx"
   --detect_model # type=str, default="runs/detect/train/weights/best.onnx"
   --cls_imgsz # type=int, default=480
   --detect_imgsz # type=int, default=1280
   ```

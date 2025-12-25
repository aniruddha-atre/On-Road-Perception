# On-Road Perception using Deep Learning

This project focuses on visual perception for autonomous driving, covering two core tasks:

2D Object Detection using YOLOv8

Semantic Segmentation using Transformer-based SegFormer

The emphasis is on model training quality, evaluation metrics, and visual results, following practices used in real-world autonomous driving pipelines.

## 1️⃣ Object Detection — YOLOv8

Trained on road-scene style set-up - KITTI dataset

Detects vehicles, pedestrians, cyclists, and traffic-relevant objects

Training & Validation Curves

Placeholder:
📊 YOLOv8 – Train / Validation Loss

<img width="1172" height="703" alt="image" src="https://github.com/user-attachments/assets/7f203cca-011c-4c7a-8ca6-4752903022ee" />

📊 YOLOv8 – mAP@50 / mAP@50-95

<img width="782" height="347" alt="image" src="https://github.com/user-attachments/assets/d133f16b-3d80-476e-8c53-d0ad0dc86f7f" />


2️⃣ Semantic Segmentation — SegFormer

Trained on Cityscapes

Pixel-wise classification into 19 semantic classes



📈 Key Results (Segmentation)
Metric	Value
Best Validation mIoU	~74%
Training Resolution	512 × 512
Validation	Full-resolution sliding window


Training Curves


📊 Train Loss vs Epoch
📊 Validation mIoU vs Epoch

<img width="878" height="411" alt="image" src="https://github.com/user-attachments/assets/7368080e-304e-43ee-ac4b-8d30ee2a6cf9" />


## 🖼 Qualitative Results

Object Detection

![object_detection (1)](https://github.com/user-attachments/assets/5d673cee-7e75-4845-b772-a94bbaf17a6f)

Semantic Segmentation

![semantic_segmentation (1)](https://github.com/user-attachments/assets/0e16f9f7-4f6b-4192-be37-df59a9c56a5b)


## Credit

1) YoloV8
2) SSegformer
3) Pexels Dashcam video


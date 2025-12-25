# On-Road Perception using Deep Learning

This project focuses on visual perception for autonomous driving, covering two core tasks:

a) 2D Object Detection using YOLOv8  
b) Semantic Segmentation using Transformer-based SegFormer

The emphasis is on model training quality, evaluation metrics, and visual results, following practices used in real-world autonomous driving pipelines.

## 🖼 Qualitative Results

Object Detection vs Semantic Segmentation

![object_detection (1)](https://github.com/user-attachments/assets/5d673cee-7e75-4845-b772-a94bbaf17a6f) &nbsp;&nbsp;&nbsp;&nbsp; ![semantic_segmentation (1)](https://github.com/user-attachments/assets/0e16f9f7-4f6b-4192-be37-df59a9c56a5b)


## 1️⃣ Object Detection — YOLOv8

Trained on road-scene style set-up - KITTI dataset

Detects vehicles, pedestrians, cyclists, and traffic-relevant objects  

📈 Key Results  
Achieved 94% mAP50
Achieved 75% mAP50-95

<br>

📊 Training & Validation Curves

a) YOLOv8 – Train / Validation Loss

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/7f203cca-011c-4c7a-8ca6-4752903022ee" /><br>

b) YOLOv8 – mAP@50 / mAP@50-95

<img width="720" height="300" alt="image" src="https://github.com/user-attachments/assets/d133f16b-3d80-476e-8c53-d0ad0dc86f7f" /><br>



## 2️⃣ Semantic Segmentation — SegFormer

Trained on Cityscapes

Pixel-wise classification into 19 semantic classes


📈 Key Results  
Best Validation mIoU:	~74%  
Training Resolution:	512 × 512  
Validation:	Full-resolution sliding window  

<br>

📊 Training Curves  


a) Train Loss vs Epoch  
b) Validation mIoU vs Epoch

<img width="720" height="300" alt="image" src="https://github.com/user-attachments/assets/7368080e-304e-43ee-ac4b-8d30ee2a6cf9" /><br>


## Credit

1) YOLOv8 (Object Detection)
Implemented using Ultralytics YOLOv8, a state-of-the-art real-time object detection framework.
https://github.com/ultralytics/ultralytics

2) SegFormer (Semantic Segmentation)
Semantic segmentation powered by NVIDIA SegFormer, a Transformer-based architecture for efficient semantic segmentation.
https://github.com/NVlabs/SegFormer

3) Dashcam Video
Sample driving video sourced from Pexels, used strictly for non-commercial, educational, and demonstration purposes.
https://www.pexels.com

# Computer Vision – Image Processing, Face Detection & Object Detection

<p align="center">
  <a href="https://drive.google.com/file/d/1J0ZXIfYKPXJzbyhVkEcxYwEpd8OUjhrC/view?usp=sharing">
    <img src="https://img.shields.io/badge/Watch%20Project%20Explanation-Google%20Drive-red?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Project Explanation">
  </a>
</p>

<p align="center">
  **Computer Vision | OpenCV | YuNet | YOLOv8 | Image Processing | Real-Time Detection**
</p>

<p align="center">
  An end-to-end Computer Vision project covering image processing, morphological operations, bitwise image manipulation, face detection, object detection, and an integrated real-time detection pipeline.
</p>

---

## 1. Project Overview

This project explores fundamental and advanced Computer Vision techniques using Python and OpenCV.

The primary objective is to understand how digital images can be processed, enhanced, analyzed, and interpreted using classical image processing techniques and deep learning-based detection models.

The project is organized into five major tasks, progressing from basic image manipulation to an integrated real-time detection system.

It begins with morphological operations, binary image processing, bitwise operations, image histograms, and brightness and contrast adjustment.

The next stage introduces **YuNet**, a lightweight deep learning-based face detector that identifies faces, facial landmarks, and confidence scores.

For general object detection, the project implements **YOLOv8n**, a pretrained object detection model capable of identifying multiple object categories using COCO-trained weights.

Finally, YuNet and YOLOv8 are integrated into a single webcam pipeline to perform face and object detection simultaneously, with FPS benchmarking to compare individual and combined inference performance.

The project combines classical Computer Vision techniques with modern deep learning-based detection models.

## 2. Project Objectives

- Understand digital image processing using OpenCV.
- Perform image loading, conversion, and visualization.
- Apply image thresholding and binarization.
- Implement erosion, dilation, opening, and closing.
- Compare morphological kernel shapes and sizes.
- Perform bitwise AND, OR, XOR, and NOT operations.
- Extract regions of interest using masks.
- Analyze grayscale and color-channel histograms.
- Adjust image brightness and contrast.
- Understand image intensity distributions.
- Implement face detection using YuNet.
- Detect facial landmarks and confidence scores.
- Experiment with face detection thresholds.
- Apply privacy-preserving face blurring.
- Implement object detection using YOLOv8n.
- Experiment with confidence and IoU thresholds.
- Perform real-time webcam detection.
- Integrate face and object detection into one pipeline.
- Measure FPS for individual and combined detection systems.

## 3. Project Architecture

The notebook is divided into five major tasks.

| Task | Topic | Main Technologies |
|---|---|---|
| Task 1 | Image Processing & Morphological Operations | OpenCV, NumPy |
| Task 2 | Bitwise Operations & Image Histograms | OpenCV, Matplotlib |
| Task 3 | Face Detection | YuNet, OpenCV |
| Task 4 | Object Detection | YOLOv8, Ultralytics |
| Task 5 | Integrated Detection Pipeline | YuNet + YOLOv8 |

---

## 4. Task 1 – Environment Setup, Image I/O & Morphological Operations

The first task focuses on understanding image representation and applying classical morphological operations.

### 4.1 Environment Setup

The project uses OpenCV and supporting Python libraries.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from collections import Counter
```

The notebook creates separate directories for downloaded images, model weights, and generated visualizations.

```text
data/
├── images/
└── models/

plots/
```

### 4.2 Image Loading

The project loads sample images using OpenCV.

The sample images include:

- `baboon.jpg`
- `smarties.png`
- `bus.jpg`
- `zidane.jpg`

OpenCV supports reading images in color and grayscale formats.

```python
img_colour = cv2.imread(
    "data/images/baboon.jpg",
    cv2.IMREAD_COLOR
)

img_gray = cv2.imread(
    "data/images/smarties.png",
    cv2.IMREAD_GRAYSCALE
)
```

OpenCV loads color images in BGR format rather than RGB. The notebook converts BGR images to RGB when displaying them using Matplotlib.

### 4.3 Image Binarization

Binarization converts a grayscale image into a black-and-white image using a threshold.

The notebook uses a threshold value of 127.

```python
gray = cv2.cvtColor(
    img_colour,
    cv2.COLOR_BGR2GRAY
)

_, binary_img = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)
```

Pixels above the threshold become white, while pixels below or equal to the threshold become black.

This operation is useful for separating foreground regions from background regions in suitable images.

### 4.4 Morphological Operations

Morphological operations process binary images using a structuring element called a kernel.

The project implements four fundamental operations.

**Erosion**

Erosion reduces the size of foreground white regions.

Applications:
- Removing small white noise.
- Separating connected foreground regions.
- Shrinking object boundaries.

**Dilation**

Dilation expands foreground white regions.

Applications:
- Connecting broken regions.
- Filling small gaps.
- Increasing foreground object size.

**Opening**

Opening performs erosion followed by dilation.

```python
opened = cv2.morphologyEx(
    binary_img,
    cv2.MORPH_OPEN,
    kernel
)
```

Applications:
- Removing small foreground noise.
- Cleaning binary masks.
- Preserving the general shape of larger objects.

**Closing**

Closing performs dilation followed by erosion.

```python
closed = cv2.morphologyEx(
    binary_img,
    cv2.MORPH_CLOSE,
    kernel
)
```

Applications:
- Filling small holes.
- Closing gaps in foreground regions.
- Repairing broken binary masks.

### 4.5 Kernel Shape and Size Comparison

The notebook compares three structuring element shapes:

- RECT
- ELLIPSE
- CROSS

It also compares three kernel sizes:

- 3×3
- 5×5
- 9×9

Both Opening and Closing are evaluated using these combinations.

This experiment demonstrates how kernel geometry and size influence noise removal, object boundaries, and the preservation of image details.

<details>
<summary><strong>Morphological Operations Summary</strong></summary>

| Operation | Main Effect | Typical Application |
|---|---|---|
| Erosion | Shrinks foreground | Noise removal |
| Dilation | Expands foreground | Gap filling |
| Opening | Removes small foreground noise | Mask cleaning |
| Closing | Fills small holes | Mask repair |

</details>

---

## 5. Task 2 – Bitwise Operations & Image Histograms

This task explores logical operations on binary masks and statistical analysis of image intensity values.

### 5.1 Bitwise Operations

The notebook creates two binary masks:

- Circular mask
- Rectangular mask

It then applies four bitwise operations.

**Bitwise AND**

Retains the overlapping white regions of both masks.

```python
result = cv2.bitwise_and(mask_a, mask_b)
```

**Bitwise OR**

Combines the white regions of both masks.

```python
result = cv2.bitwise_or(mask_a, mask_b)
```

**Bitwise XOR**

Retains regions where the masks differ.

```python
result = cv2.bitwise_xor(mask_a, mask_b)
```

**Bitwise NOT**

Inverts the pixel values of a mask.

```python
result = cv2.bitwise_not(mask_a)
```

### 5.2 Applied Use Case – Masked Overlay

Bitwise operations are also used for practical image manipulation.

The notebook demonstrates:

- Extracting a circular region from a photograph.
- Creating a region-of-interest mask.
- Applying the mask using Bitwise AND.
- Creating a simple logo.
- Combining the logo with an image using Bitwise OR.

This demonstrates how masks can be used to isolate specific image regions and combine visual elements.

### 5.3 Grayscale Histogram

A grayscale histogram represents the distribution of pixel intensity values.

The intensity range is:

```text
0 = Black
255 = White
```

The notebook uses OpenCV's `calcHist()` function:

```python
gray_hist = cv2.calcHist(
    [img_gray],
    [0],
    None,
    [256],
    [0, 256]
)
```

Histograms help identify:

- Dark and bright image regions.
- Overall image exposure.
- Contrast distribution.
- Intensity clipping.

### 5.4 Color Channel Histograms

The notebook separately plots the Blue, Green, and Red channel histograms.

OpenCV stores color channels in BGR order.

This analysis helps understand how pixel intensities are distributed across different color channels.

### 5.5 Brightness and Contrast Adjustment

Brightness and contrast are modified using OpenCV's `convertScaleAbs()`.

The mathematical relationship is:

```text
new_pixel = alpha × pixel + beta
```

Where:

- `alpha` controls contrast.
- `beta` controls brightness.

The notebook compares the original image with two modified versions.

**Brighter / Higher Contrast**

```python
brightened = cv2.convertScaleAbs(
    img_colour,
    alpha=1.5,
    beta=30
)
```

**Darker / Lower Contrast**

```python
darkened = cv2.convertScaleAbs(
    img_colour,
    alpha=0.7,
    beta=-30
)
```

The corresponding grayscale histograms are also plotted to visualize the changes in intensity distribution.

### Practical Applications

- Image enhancement.
- Preprocessing before detection.
- Exposure analysis.
- Image quality inspection.
- Visual analysis of low-contrast images.

---

## 6. Task 3 – Face Detection using YuNet

This task implements deep learning-based face detection using **YuNet** through OpenCV's `FaceDetectorYN` API.

YuNet is a lightweight face detector designed for efficient face detection.

It returns:

- Face bounding boxes.
- Five facial landmarks.
- Detection confidence scores.

### 6.1 YuNet Model Loading

The notebook downloads the pretrained ONNX model:

```text
face_detection_yunet_2023mar.onnx
```

Model source:

```text
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The model is saved inside:

```text
data/models/
```

The notebook checks whether OpenCV supports:

```python
cv2.FaceDetectorYN_create
```

### 6.2 Static Image Face Detection

YuNet is applied to the sample images:

- `zidane.jpg`
- `baboon.jpg`
- `bus.jpg`

For each detected face, the output includes:

**Bounding Box**

A rectangular box around the detected face.

**Facial Landmarks**

Five landmark positions representing facial key points.

**Confidence Score**

A numerical score representing the detector's confidence in the face detection.

### 6.3 Face Detection Visualization

The notebook draws:

- Green bounding boxes.
- Five colored landmark points.
- Confidence values above detected faces.

This makes it easier to inspect detection results visually.

### 6.4 Real-Time YuNet Webcam Detection

A webcam function is implemented for real-time face detection.

```python
run_yunet_webcam()
```

The function:

1. Opens the webcam.
2. Reads frames continuously.
3. Initializes the YuNet detector.
4. Detects faces in each frame.
5. Draws bounding boxes and landmarks.
6. Displays the current FPS.
7. Stops when the user presses `q`.

The function is optional and should be executed locally because notebook environments may not provide direct webcam access.

### 6.5 Confidence Threshold Experiment

The notebook compares three YuNet confidence thresholds:

```python
[0.5, 0.7, 0.9]
```

A lower threshold generally accepts more detection candidates, potentially increasing false positives.

A higher threshold is more conservative and may miss difficult faces.

The experiment records the number of detected faces at each threshold and visualizes the results.

### 6.6 Privacy Blur

The project also demonstrates privacy-preserving face blurring.

After detecting faces, the corresponding face regions are blurred using Gaussian Blur.

```python
cv2.GaussianBlur(
    face_region,
    (31, 31),
    0
)
```

This demonstrates how face detection can support privacy-oriented image processing.

### Applications of YuNet

- Face detection in images.
- Webcam-based face detection.
- Face localization.
- Facial landmark visualization.
- Privacy masking.
- Lightweight computer vision applications.

---

## 7. Task 4 – Object Detection using YOLOv8

This task implements object detection using **YOLOv8 Nano (YOLOv8n)** from the Ultralytics library.

YOLO is a single-stage object detector that predicts object bounding boxes and class probabilities in a forward pass.

The pretrained YOLOv8n model uses COCO-trained weights and supports detection across 80 object categories.

### 7.1 Loading YOLOv8n

The model is loaded using:

```python
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")
```

The pretrained weights are downloaded automatically when required.

YOLOv8n is selected because it prioritizes speed and relatively low computational requirements.

### 7.2 Static Image Object Detection

The model is evaluated on:

- `bus.jpg`
- `zidane.jpg`
- `smarties.png`

The model detects objects and generates:

- Bounding boxes.
- Class labels.
- Confidence scores.

The Ultralytics `plot()` method is used to visualize the detection results.

```python
result = yolo_model(
    "data/images/bus.jpg",
    verbose=False
)[0]

annotated_image = result.plot()
```

### 7.3 Real-Time YOLO Webcam Detection

The project includes a webcam-based object detection function:

```python
run_yolo_webcam()
```

The function:

1. Opens the webcam.
2. Captures video frames.
3. Runs YOLOv8 inference.
4. Draws object detection results.
5. Displays the current FPS.
6. Stops when the user presses `q`.

### 7.4 Confidence Threshold

The confidence threshold controls the minimum confidence required for a detection to be retained.

The notebook compares:

```python
[0.25, 0.5, 0.75]
```

A lower confidence threshold may retain more candidate detections, including uncertain predictions.

A higher threshold reduces the number of accepted detections but may remove valid objects.

### 7.5 IoU Threshold and Non-Maximum Suppression

Intersection over Union (IoU) measures the overlap between two bounding boxes.

```text
IoU = Area of Intersection / Area of Union
```

The notebook compares:

```python
[0.3, 0.5, 0.7]
```

YOLO uses Non-Maximum Suppression (NMS) to reduce duplicate bounding boxes.

The IoU threshold influences how overlapping detections are handled.

### 7.6 Per-Class Detection Summary

The notebook collects detection counts for the detected COCO classes across the sample images.

A horizontal bar chart is generated to visualize the number of detections per class.

This helps understand the distribution of detected objects in the selected sample images.

### Applications of YOLOv8

- Object detection in images.
- Real-time video analytics.
- Traffic monitoring.
- Object localization.
- Industrial inspection.
- Smart surveillance.
- Automated visual recognition.

---

## 8. Task 5 – Integrated Face and Object Detection Pipeline

The final task combines YuNet and YOLOv8 into a unified real-time webcam pipeline.

The goal is to detect faces and general objects simultaneously.

### 8.1 Integrated Pipeline Architecture

```text
                  Webcam Input
                       |
                       v
                  Capture Frame
                       |
                       v
              Optional Preprocessing
                       |
             +---------+---------+
             |                   |
             v                   v
         YOLOv8n               YuNet
             |                   |
             v                   v
       Object Detection      Face Detection
             |                   |
             v                   v
      Object Bounding       Face Bounding
       Boxes and Labels     Boxes and Scores
             |                   |
             +---------+---------+
                       |
                       v
               Combined Output
                       |
                       v
                  FPS Display
                       |
                       v
                 Live Webcam
```

### 8.2 Unified Webcam Function

The integrated pipeline is implemented through:

```python
run_integrated_pipeline()
```

It performs the following operations:

1. Captures webcam frames.
2. Optionally applies morphological preprocessing.
3. Runs YOLOv8 object detection.
4. Runs YuNet face detection.
5. Draws object labels and bounding boxes.
6. Draws face bounding boxes and confidence scores.
7. Displays FPS.
8. Continues until the user presses `q`.

### 8.3 Detection Visualization

The integrated pipeline uses:

| Detector | Visualization |
|---|---|
| YOLOv8 | Object bounding boxes and class labels |
| YuNet | Green face bounding boxes and confidence |
| FPS | Live frame-rate information |

This allows both detection systems to operate on the same webcam stream.

### 8.4 Morphological Pre-Cleaning

The notebook explores morphological preprocessing before object detection.

The experiment includes:

1. Converting the frame to grayscale.
2. Applying binary thresholding.
3. Creating a threshold mask.
4. Applying morphological Opening.
5. Comparing the original mask with the cleaned mask.

The integrated function supports:

```python
run_integrated_pipeline(
    morphology_preclean=True
)
```

The default is:

```python
morphology_preclean=False
```

**Important:** Thresholding and masking a color frame can remove useful image information. Therefore, morphological preprocessing should only be enabled if actual detection results improve.

### 8.5 FPS Benchmarking

The notebook implements a benchmarking function:

```python
benchmark_pipeline_fps()
```

It captures up to 100 webcam frames and measures inference performance for:

- YuNet face detection.
- YOLOv8 object detection.
- Combined YuNet and YOLOv8 inference.

The FPS is calculated as:

```text
FPS = Number of Processed Frames / Total Processing Time
```

A bar chart is generated to compare the measured FPS values.

FPS depends on:

- CPU or GPU hardware.
- Webcam resolution.
- Model inference time.
- OpenCV configuration.
- YOLO model size.
- System load.

The notebook intentionally measures FPS on the user's own hardware rather than assuming fixed performance values.

---

## 9. Complete Feature Summary

| Feature | Implementation |
|---|---|
| Image Loading | OpenCV |
| Grayscale Conversion | OpenCV |
| Binary Thresholding | OpenCV |
| Erosion | Morphological Operations |
| Dilation | Morphological Operations |
| Opening | Morphological Operations |
| Closing | Morphological Operations |
| Kernel Comparison | RECT, ELLIPSE, CROSS |
| Bitwise Operations | AND, OR, XOR, NOT |
| Region of Interest | Binary Masking |
| Image Histograms | Grayscale and BGR |
| Brightness Adjustment | Alpha-Beta Transformation |
| Face Detection | YuNet ONNX |
| Facial Landmarks | YuNet |
| Privacy Blur | Gaussian Blur |
| Object Detection | YOLOv8n |
| Confidence Tuning | YOLOv8 |
| IoU Tuning | YOLOv8 NMS |
| Webcam Detection | OpenCV VideoCapture |
| Integrated Pipeline | YuNet + YOLOv8 |
| Performance Benchmark | FPS Measurement |

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and detection |
| NumPy | Array and numerical operations |
| Matplotlib | Image and histogram visualization |
| Pandas | Detection comparison tables |
| SciPy | Manual convolution |
| Ultralytics | YOLOv8 object detection |
| YuNet | Lightweight face detection |
| ONNX | YuNet pretrained model format |
| Jupyter Notebook | Interactive development |
| Google Colab | Notebook execution environment |

## 11. Project Structure

The notebook automatically creates directories for images, model files, and generated plots.

```text
Computer-Vision-PR3/
│
├── CV_PR3.ipynb
├── README.md
│
├── data/
│   ├── images/
│   │   ├── baboon.jpg
│   │   ├── smarties.png
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   │
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
│
└── plots/
    ├── morphology_grid.png
    ├── opening_kernel_grid.png
    ├── closing_kernel_grid.png
    ├── bitwise_grid.png
    ├── masked_overlay.png
    ├── grayscale_histogram.png
    ├── colour_histograms.png
    ├── histogram_comparison.png
    ├── yolo_detections.png
    └── fps_comparison.png
```

The notebook downloads the sample images and YuNet model into the corresponding directories when required.

The `plots/` directory stores generated visualizations.

## 12. Installation and Setup

### Step 1: Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

The notebook specifies the following dependencies:

```bash
pip install opencv-python ultralytics matplotlib numpy pandas scipy
```

For YuNet support, use an OpenCV build that provides:

```python
cv2.FaceDetectorYN_create
```

The notebook references OpenCV 5.0.0 packages in its setup instructions. If that version is unavailable for your Python version or platform, install a compatible OpenCV build and verify the API.

Avoid installing conflicting `opencv-python` and `opencv-contrib-python` versions together. Use a single compatible OpenCV distribution.

### Step 4: Open Jupyter Notebook

```bash
pip install jupyter
jupyter notebook
```

Open:

```text
CV_PR3.ipynb
```

### Step 5: Execute the Notebook

Run the cells sequentially.

The notebook automatically:

- Creates the required directories.
- Downloads sample images.
- Downloads the YuNet ONNX model.
- Loads YOLOv8n weights when required.
- Generates visualizations.
- Saves selected plots in the `plots/` directory.

### Step 6: Run Webcam Functions Locally

Execute the following functions in a local Python or notebook environment:

```python
run_yunet_webcam()
```

```python
run_yolo_webcam()
```

```python
run_integrated_pipeline()
```

To benchmark all available pipelines:

```python
benchmark_pipeline_fps()
```

Press `q` in the OpenCV display window to stop webcam detection.

---

## 13. Important Parameters

### YuNet Configuration

| Parameter | Value |
|---|---:|
| Score Threshold | 0.5 / 0.7 / 0.9 |
| NMS Threshold | 0.3 |
| Top K | 5000 |
| Model Format | ONNX |

### YOLOv8 Configuration

| Parameter | Values Tested |
|---|---|
| Model | YOLOv8n |
| Confidence Threshold | 0.25, 0.5, 0.75 |
| IoU Threshold | 0.3, 0.5, 0.7 |
| Pretrained Dataset | COCO |
| Model Type | Single-stage detector |

### Morphological Configuration

| Parameter | Values |
|---|---|
| Threshold | 127 |
| Main Kernel | 5×5 |
| Kernel Shapes | RECT, ELLIPSE, CROSS |
| Kernel Sizes | 3×3, 5×5, 9×9 |
| Operations | Erosion, Dilation, Opening, Closing |

## 14. Practical Applications

The techniques explored in this project have applications across several Computer Vision domains.

**Smart Surveillance**

Combining face and object detection can help analyze live video streams.

**Privacy Protection**

YuNet face detection can be used to locate faces before applying privacy blur.

**Traffic Monitoring**

YOLO can detect vehicles and other objects in traffic scenes.

**Industrial Inspection**

Morphological operations can help clean binary masks and isolate regions of interest.

**Image Enhancement**

Histograms and brightness/contrast adjustment can help analyze and improve image quality.

**Real-Time Video Analytics**

Combining detection models provides a foundation for real-time visual monitoring systems.

## 15. Limitations

- Face detection depends on lighting, facial visibility, and camera quality.
- YuNet may miss faces under occlusion or unusual viewpoints.
- YOLOv8n may struggle with small or partially hidden objects.
- YOLOv8n is limited to the categories supported by its pretrained COCO weights.
- Detection confidence thresholds affect precision and recall.
- Morphological preprocessing may remove useful image details if applied incorrectly.
- Webcam performance varies across hardware.
- The integrated pipeline may run slower than either detector independently.
- Webcam functions require a locally accessible camera.
- FPS results must be measured on the actual target device.

## 16. Future Improvements

Potential improvements include:

- Add object tracking using ByteTrack or DeepSORT.
- Add persistent object IDs across video frames.
- Implement face recognition with appropriate consent and safeguards.
- Add configurable detection thresholds through a user interface.
- Compare YOLOv8n with larger YOLO architectures.
- Add GPU acceleration and inference optimization.
- Implement video file upload and processing.
- Save annotated output videos.
- Add real-time detection statistics.
- Develop a Streamlit-based Computer Vision dashboard.
- Deploy the detection API using FastAPI.
- Add automated performance testing.
- Evaluate detectors using a dedicated benchmark dataset.

## 17. Key Learnings

This project provides practical experience with:

- Digital image processing.
- Image thresholding and binarization.
- Morphological image operations.
- Structuring elements and kernel design.
- Bitwise image manipulation.
- Region-of-interest extraction.
- Grayscale and color histograms.
- Brightness and contrast transformations.
- Deep learning-based face detection.
- Facial landmark localization.
- ONNX model integration.
- YOLOv8 object detection.
- Confidence and IoU threshold tuning.
- Non-Maximum Suppression.
- Real-time webcam processing.
- Integrated Computer Vision pipelines.
- FPS measurement and performance comparison.

## 18. Conclusion

This Computer Vision project demonstrates the integration of classical image processing techniques with modern deep learning-based detection models.

The first two tasks establish the fundamentals of image manipulation through morphological operations, bitwise processing, histograms, and intensity transformations.

The next stages introduce YuNet for face detection and YOLOv8n for general object detection.

Finally, both detectors are integrated into a unified webcam pipeline with optional morphological preprocessing and FPS benchmarking.

The project demonstrates practical knowledge of **OpenCV, image processing, YuNet, YOLOv8, real-time video analytics, and Computer Vision system integration**.

---

## Project Explanation Video

Watch the complete project explanation and implementation:

[**Watch Computer Vision Project – Google Drive**](https://drive.google.com/file/d/1J0ZXIfYKPXJzbyhVkEcxYwEpd8OUjhrC/view?usp=sharing)

## Author

**Parth Solanki**

BCA (Hons) Student | Machine Learning | Deep Learning | Computer Vision | Generative AI

## License

This project is intended for educational, academic, and portfolio purposes. Add a formal license file if you wish to define reuse permissions.

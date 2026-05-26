# Real-Time Object Detection and Measurement with OpenCV and ArUco Markers

A real-time Machine Vision project that detects, measures, and classifies objects from a live camera stream using **Python**, **OpenCV**, and an **ArUco marker** as a physical scale reference.

The application compares three contour-extraction strategies—**Canny Edge Detection**, **Global Binarization**, and **Adaptive Thresholding**—and converts detected object dimensions from pixels to centimeters in real time. In the recorded demonstration, **Adaptive Thresholding** was selected because it produced the most reliable qualitative results under the tested setup.

---

## Demo

The repository includes a recorded live demonstration:

**[Watch the video demonstration](./VideoDemonstration.mp4)**

In the demonstration, the system:

- detects a `10 cm` ArUco reference marker;
- computes and overlays the pixel-to-centimeter ratio;
- identifies object contours in a live camera feed;
- displays measured dimensions and classification labels for recognized objects.

> **Performance note:** the choice of Adaptive Thresholding is based on qualitative real-time observations from the demonstration environment. The current project does not include a labeled test dataset or quantitative benchmarks such as accuracy, mean measurement error, or inference FPS.

---

## Features

- Real-time video acquisition from an external camera.
- ArUco-based metric calibration using a known-size marker.
- Three selectable image-processing pipelines:
  - Canny Edge Detection;
  - Global Binarization;
  - Adaptive Thresholding.
- Contour extraction and bounding-box generation.
- Real-world dimensional measurement in centimeters.
- Rule-based classification of:
  - credit cards;
  - AAA batteries.
- Live visualization of marker detection, calibration ratio, bounding boxes, labels, and measured dimensions.

---

## System Pipeline

```text
Camera Frame
     |
     v
Grayscale Conversion
     |
     v
ArUco Marker Detection ----> Pixel-to-Centimeter Calibration
     |
     v
Selected Preprocessing Method
(Canny / Binarization / Adaptive Thresholding)
     |
     v
Contour Extraction
     |
     v
Bounding Box Measurement in Pixels
     |
     v
Conversion to Centimeters
     |
     v
Dimension-Based Object Classification
     |
     v
Annotated Real-Time Output
```

---

## How Calibration Works

A printed ArUco marker with known side length is placed in the camera scene. The application detects its four corners and estimates the scale factor as:

```text
pixel_to_cm_ratio = average(marker_width_pixels, marker_height_pixels) / marker_size_cm
```

The current implementation uses:

- **ArUco dictionary:** `DICT_4X4_100`
- **Marker side length:** `10.0 cm`

Averaging marker width and height helps reduce the impact of minor perspective distortion or non-ideal camera positioning. Once the ratio is available, every detected contour can be measured in centimeters.

---

## Detection Methods

All three methods first convert the incoming frame to grayscale and apply a `5 x 5` Gaussian blur before extracting contours.

| Method | Processing Step | Parameters | Notes |
|---|---|---|---|
| Canny Edge Detection | Detects object edges from intensity gradients | User-selected lower and upper thresholds | Useful when boundaries are sharp and lighting is controlled |
| Global Binarization | Applies inverse binary thresholding | User-selected threshold | Simple and fast, but sensitive to lighting changes |
| Adaptive Thresholding | Applies local Gaussian adaptive thresholding | `block_size=11`, `C=2` | Selected for the recorded demo because it gave the best qualitative results in the tested conditions |

---

## Object Classification

The project classifies objects by comparing their measured bounding-box dimensions with predefined real-world ranges.

| Object | Accepted Length Range | Accepted Height Range |
|---|---:|---:|
| Credit Card | `8.5 – 9.5 cm` | `5.0 – 6.0 cm` |
| AAA Battery | `4.0 – 5.0 cm` | `0.5 – 1.5 cm` |

This is a lightweight rule-based classifier: it is easy to inspect and fast to execute, but it assumes that objects are visible from above, sufficiently separated from the background, and approximately aligned with the camera plane.

---

## Repository Structure

```text
.
├── main.py                    # Live camera loop, ArUco calibration, measurement and classification
├── MyDetectionMethods.py      # Canny, binarization and adaptive-thresholding pipelines
├── VideoDemonstration.mp4     # Recorded real-time demonstration
└── README.md                  # Project documentation
```

---

## Requirements

- Python 3
- A webcam or external camera
- A printed `DICT_4X4_100` ArUco marker with a measured side length of `10 cm`

Python dependencies:

```bash
pip install numpy opencv-contrib-python
```

`opencv-contrib-python` is required because the project uses OpenCV's `aruco` module.

---

## Running the Project

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/accorintibenito/real-time-object-detection-and-measurement-with-OpenCV-and-ArUco-markers.git
   cd real-time-object-detection-and-measurement-with-OpenCV-and-ArUco-markers
   ```

2. Install dependencies:

   ```bash
   pip install numpy opencv-contrib-python
   ```

3. Place a printed `10 cm x 10 cm` ArUco marker in the camera view.

4. Run the application:

   ```bash
   python main.py
   ```

5. Select a filtering method from the console:

   ```text
   1. Canny Filter
   2. Binarization
   3. Adaptive Thresholding
   ```

6. Press `q` in the OpenCV window to stop the application.

---

## Camera Configuration

The current source code initializes the camera using:

```python
cap = cv2.VideoCapture(1)
```

This assumes that the desired external camera is exposed by the operating system as device index `1`. When using the built-in camera or a different setup, update the index in `main.py`, for example:

```python
cap = cv2.VideoCapture(0)
```

---

## Results and Method Comparison

Three preprocessing strategies were implemented and tested in the same real-time measurement workflow.

| Method | Main Strength | Main Limitation in This Use Case |
|---|---|---|
| Canny Edge Detection | Provides explicit object-edge extraction | Requires threshold tuning and may generate fragmented contours |
| Global Binarization | Straightforward segmentation pipeline | A fixed threshold can be sensitive to non-uniform illumination |
| Adaptive Thresholding | Handles local lighting variations more robustly | May require tuning for noisy backgrounds or different object textures |

For the recorded demonstration, **Adaptive Thresholding** was used because it returned the best visual detection results in the tested camera and lighting conditions.

### Current Evaluation Scope

The current repository demonstrates functional, real-time object measurement and classification. It does **not** yet claim:

- dataset-based classification accuracy;
- measured precision or recall;
- statistically validated dimensional error;
- benchmarked frames-per-second performance.

These metrics are natural next steps for extending the project into a more rigorous computer-vision evaluation.

---

## Limitations

- Classification is based only on bounding-box dimensions rather than learned visual features.
- Measurement accuracy depends on marker visibility, camera angle, lens distortion, lighting, and contour quality.
- Rotated objects may produce less accurate axis-aligned bounding-box measurements.
- The scene must include a detectable ArUco marker before object dimensions can be estimated.
- Threshold parameters may need adjustment when the background or illumination changes.

---

## Potential Improvements

- Add rotated minimum-area rectangles for more robust measurements of angled objects.
- Calibrate the camera to correct lens distortion.
- Record quantitative error against ground-truth object measurements.
- Measure FPS and detection stability across lighting/background conditions.
- Add automatic threshold tuning and contour filtering by area or aspect ratio.
- Extend classification to additional objects or replace dimensional rules with a learned detector.
- Refactor repeated classification and drawing logic into reusable functions.

---

## Technologies Used

- **Python**
- **OpenCV**
- **OpenCV ArUco module**
- **NumPy**

---

## Author

Developed as a Machine Vision University project focused on real-time image processing, metric calibration, contour-based measurement, and interpretable object classification.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

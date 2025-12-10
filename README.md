
# Detect, Align, Crop (RFB-ONNX)

**A high-throughput, edge-optimized subsystem for face detection and geometric normalization.**

This repository implements a lightweight computer vision pipeline designed to prepare facial imagery for downstream biometric recognition tasks. It leverages an **RFB (Receptive Field Block)** detector for scale-invariant face localization and a specialized **3-point landmark regressor** for affine alignment, running entirely within the **ONNX Runtime** environment for hardware-agnostic deployment.

-----

## 🏗 System Architecture

The pipeline operates as a directed acyclic graph (DAG) consisting of two distinct stages. The design prioritizes low latency and minimal memory footprint (\<5MB total model weight).

### **Stage 1: Face Localization (RFB-Net)**

  * **Backbone:** Reduced-capacity RFB module optimized for feature extractability on low-power devices.
  * **Decoding:** Custom implementations (`detection_output_registration.py`, `prior_box_registration.py`) handle anchor generation and Non-Maximum Suppression (NMS) externally to the graph, allowing for flexible threshold tuning without re-exporting models.
  * **Output:** Raw bounding boxes and confidence scores.

### **Stage 2: Geometric Normalization (3-Point Alignment)**

  * **Landmark Strategy:** Unlike standard 5-point schemas, this system utilizes a **3-point topology (Inner Left Eye, Inner Right Eye, Bottom Lip)**. This triangular configuration offers superior stability for vertical alignment and reduces jitter in yaw-heavy poses.
  * **Transformation:** A similarity transformation matrix is calculated to map the source face to a canonical template, ensuring consistent ocular and mouth positioning for downstream feature extractors.

-----

## 📂 Repository Map

The codebase is structured to separate model artifacts from processing logic.

```text
Tanmay-FF/detect_align_crop_rfb/
├── onnx_models/                    # Serialized Inference Graphs
│   ├── model_with_output_v5.onnx   # Face Detector (RFB variant) [1.1 MB]
│   └── landmark_model.onnx         # 3-Point Regressor [3.8 MB]
├── detect_align_crop.py            # Pipeline Entry Point (Orchestrator)
├── bounding_box.py                 # Geometry Utilities & NMS Implementation
├── detection_output_registration.py# Detector Output Decoding Layer
├── prior_box_registration.py       # Anchor/Prior Box Generator
└── requirements.txt                # Dependency Manifest
```

-----

## ⚡ Execution

The system is encapsulated in `detect_align_crop.py`. It requires the input data directory and explicit paths to the computational graphs (ONNX models).

### **CLI Command**

```bash
python detect_align_crop.py \
    --image_folder "path/to/raw_images" \
    --detector onnx_models/model_with_output_v5.onnx \
    --landmark onnx_models/landmark_model.onnx \
    --output_folder output_results
```

### **Parameters**

  * `--image_folder`: Root directory containing the raw ingestion batch.
  * `--detector`: Path to the RFB-based detection graph.
  * `--landmark`: Path to the secondary stage landmark regression graph.
  * `--output_folder` : Path to store the output which is cropped + aligned image

-----

## 📊 Technical Specifications

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **Inference Engine** | ONNX Runtime | Decoupled from Caffe/TF for portability. |
| **Total Weight** | \~4.9 MB | Extremely lightweight; suitable for IoT/Edge. |
| **Detector Input** | 360x480 | Resolution agnostic (depending on export). |
| **Landmark Input** | 64x64 | Needs resizing. |
| **Alignment Topology**| 3-Point | **[Inner-Eyes, Bottom-Lip]** configuration. |

-----

## 🔧 Deployment Notes

  * **Anchor Generation:** The `prior_box_registration.py` module generates priors dynamically. Ensure that the image input dimensions match the aspect ratios defined in the prior configuration to avoid localization drift.
  * **Concurrency:** The scripts currently execute sequentially. For high-volume production environments, it is recommended to wrap `detect_align_crop.py` in a multi-threaded producer-consumer loader.

-----


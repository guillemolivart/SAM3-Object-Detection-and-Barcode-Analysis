# SAM3 - Object Detection and Barcode Analysis
## 1\. Introduction

This repository presents a computer vision pipeline built for object detection and barcode analysis. Instead of relying on rigid, pre-trained detectors, the architecture uses the SAM3 foundation model to achieve a flexible, zero-shot solution. This allows the system to identify objects and segment barcodes without needing specific training for each new item. The workflow locates the object, extracts the barcode, and retrieves the associated metadata. You can find the full code in the `sam3_pipeline.ipynb` notebook.

---

## 2. Object Detection and Identification

### 2.1 Methodology

The system architecture was designed to operate without relying on standard pre-trained object detection models such as **YOLO, Faster R-CNN, or SSD**. This design choice necessitated a shift from traditional bounding-box regression approaches to a flexible, segmentation-based methodology capable of handling novel objects.

To implement this, the pipeline utilizes **SAM3 (Segment Anything Model 3)**.

* **Foundation Model Context:** The project integrates **SAM3**, the foundation model released by **Meta** on November 20, 2025. Representing the current **State-of-the-Art (SOTA)** in computer vision, SAM3 provides robust zero-shot capabilities. This allows the system to perform high-precision segmentation on unseen objects without the need for task-specific model training.
* **Architecture & Mechanism:** Unlike traditional CNN-based architectures, SAM3 functions as a Large Multimodal Model (LMM). It combines a high-capacity Vision Transformer (ViT) image encoder with a prompt encoder. These components project both visual features and text/geometric inputs into a shared latent space. A lightweight mask decoder then predicts segmentation masks by aligning image embeddings with prompt embeddings, effectively "understanding" and isolating objects based on semantic context.

### Comparison with State-of-the-Art (SOTA)

**SAM 3** was selected as the core engine due to its dominance in open-vocabulary benchmarks and superior generalization compared to other leading architectures. According to Meta AI's official metrics:

* **Zero-Shot Performance:** On the **SA-Co/Gold** benchmark, SAM 3 achieves a **cgF1 score of 54.1**, effectively **doubling** the performance of alternatives like **OWLv2** (24.6) and **DINO-X** (21.3).
* **Best-in-Class Segmentation:** It leads the challenging **LVIS** dataset with **48.5 AP**, surpassing OWLv2 (43.4).
* **Robust Detection:** In standard bounding box tasks (COCO), it maintains top-tier performance with **56.4 AP**, outperforming specialized detectors.

These results confirm SAM3 as the most robust engine available for discovering and segmenting novel objects in a zero-shot context.
### 2.2 Implementation

The detection process was implemented as follows:

1. **Text Prompting:** The model is queried with one expected object name (e.g., "Bottle", "Box").
2. **Mask Generation:** SAM3 generates a precise binary mask for the prompted entity.
3. **Bounding Box Extraction:** The bounding box is computationally derived from the spatial extrema of the generated mask, effectively converting segmentation outputs into detection coordinates.

### 2.3 Performance Limitation

The primary drawback of this approach was the **inference latency**. Due to the lack of a high-performance GPU, the heavy computational load of the transformer-based SAM3 architecture resulted in slow processing times per image. However, this latency is strictly a hardware constraint rather than an architectural flaw. With access to high-performance computing resources, the parallel processing capabilities of the Transformer architecture would be fully leveraged. This would drastically reduce inference time, effectively enabling the system to operate at near real-time speeds.

---

## 3\. Barcode Segmentation and Decoding

### 3.1 Barcode Segmentation

SAM3 was utilized to segment the barcodes within the detected objects. Initial experiments with generic prompts such as "barcode" proved insufficient, as the model tended to segment the entire adhesive label, including the white background. To achieve pixel-level precision, a prompt engineering strategy was employed using the highly specific query: "tight crop of black vertical barcode lines, ink pattern only". This resulted in a clean segmentation mask containing only the data-bearing elements, which was critical for minimizing noise in the subsequent geometric analysis.

### 3.2 Normal Vector Computation

To estimate the physical orientation of the barcode, the system applied **Principal Component Analysis (PCA)** to the segmentation mask's pixel coordinates.

* **Mathematical Process:** The system first constructed the **covariance matrix** of the centered pixel data to capture spatial correlation. Through **eigendecomposition**, the **eigenvalues** and **eigenvectors** were extracted.
* **Geometric Interpretation:** The resulting eigenvectors define the principal axes of the object. The vector associated with the largest eigenvalue represents the barcode's longitudinal axis (length), while the orthogonal vector allows for the estimation of the surface normal. This computation provides the precise rotation angle required for robotic alignment tasks.

### 3.3 Barcode Decoding Approaches

Two distinct methodologies were attempted to decode the content of the segmented barcodes.

#### Approach 1: Classical Image Processing
The initial attempt relied on traditional signal processing techniques, requiring a pre-processing step to geometrically normalize the input.

* **Spatial Rectification (`crop_barcode`):** Before analysis, a custom algorithm named `crop_barcode` was implemented to standardize the input. Leveraging the orientation vector calculated via PCA, the system applies an **affine transformation** to derotate the original image. This process generates a rectified **Region of Interest (ROI)** where the barcode is aligned horizontally and tightly cropped.

* **Scanline Decoding & Width Quantization (`bar_value`):** The core decoding logic is encapsulated in the `bar_value` function. This function operates on the rectified patch by extracting a horizontal scanline and measuring the raw pixel widths of the alternating black and white sequences.
    * **Adaptive Thresholding (K-Means):** To address the variability in line thickness, an unsupervised learning approach was attempted within this pipeline. A **K-Means clustering algorithm ($k=4$)** was applied to the detected widths to group them into the four standard logical sizes ($1\times, 2\times, 3\times, 4\times$) typical of barcode symbologies.
    * **Decoding Logic:** The system attempted to parse the clustered sequence in tuples of 6 elements (3 bars, 3 spaces), which constitute a single ASCII character in standards like Code128, mapping the pattern of thicknesses to the corresponding character table.

* **Performance Analysis & Limitations:** The approach proved insufficiently robust due to several critical factors:
    1.  **Inconsistent Module Widths:** The actual thickness of the printed bars varied drastically depending on the object's distance and printing quality. It was impossible to define a **global threshold**; parameters that successfully detected thick bars caused thin bars to vanish, while sensitive thresholds introduced noise artifacts.
    2.  **Clustering Ambiguity:** The "middle" widths (e.g., distinguishing between a $2\times$ and a $3\times$ width) were often blurred by anti-aliasing and low resolution, causing K-Means to misclassify boundaries and corrupt the decoding sequence.
    3.  **Geometric Distortion:** Despite rectification, cylindrical curvature (e.g., on bottles) caused non-linear width compression at the edges, which classical linear interpolation could not correct.

#### Approach 2: Deep Learning (CRNN + LSTM)

To achieve robustness against geometric distortions that baffled the classical approach, a **Convolutional Recurrent Neural Network (CRNN)** architecture was designed. The implementation and experimental code for this model are documented in the notebook `crnn_experiment.ipynb`.

* **Input Strategy (Integration with `crop_barcode`):** This deep learning pipeline was engineered to work in tandem with the spatial rectification algorithm developed in Approach 1. Instead of processing the full raw image, the network receives the **rectified Region of Interest (ROI)** generated by `crop_barcode`. This pre-processing step decouples the localization/rotation problem from the reading problem, ensuring the model receives standardized, horizontally aligned inputs.
* **CNN (Visual Feature Encoder):** The Convolutional layers act as the feature extractor. They process the raw input image to identify high-level visual patterns—specifically the gradients, edges, and width variations of the barcode lines—transforming the 2D image into a dense sequence of feature vectors.
* **LSTM (Sequential Decoder):**  The Long Short-Term Memory layers handle the sequential nature of the data. Since a barcode is read linearly, the LSTM processes the feature sequence generated by the CNN from left to right, modeling the contextual dependencies between adjacent bars to predict the final alphanumeric string.
* **Dataset Generation:** A synthetic dataset was created using the library barcode, heavily augmented with gaussian noise, random rotations, cylindrical curvature simulation and perspective warping.

* **Current Status & Computational Constraints:**
The architectural design is complete and verified. However, the full training cycle was deferred due to hardware constraints (CPU-only environment). This architecture remains a viable solution for handling complex geometric distortions given adequate computational resources.

---

## 4. Object-Barcode Spatial Association

To organize the detected elements into a coherent structure, the pipeline implements a **geometric containment logic**. Since multiple objects and barcodes may appear simultaneously, the system analyzes their spatial topology to determine ownership. If a barcode's bounding box lies within an object's segmentation mask ($B_{barcode} \subset B_{object}$), it is strictly associated with that specific object instance. This establishes a **hierarchical link**, ensuring that every detected barcode is correctly mapped to its parent object (e.g., identifying *which* specific bottle carries the detected label).

---

## 5. Architectural Specification: Information Retrieval

To bridge raw perception with high-level knowledge, the system design defines a **Bidirectional Retrieval Strategy** intended for downstream integration.

* **Data Structure Logic:** The architecture specifies a **Bidirectional Hash Map** to link unique decoded identifiers with semantic metadata. This design choice ensures **$O(1)$ time complexity**, allowing for instant retrieval regardless of inventory scale once the decoding stream is active.

* **Designed Workflows:**
    * **Identification ($Code \rightarrow Object$):** Designed to retrieve an object's semantic class (e.g., "Bottle") upon receiving a specific barcode string.
    * **Logistics ($Object \rightarrow Code$):** Designed to return all tracking identifiers associated with a specific semantic query.

* **Integration Status:** This module is currently defined as a **design specification**. The data model is ready to be populated as soon as the upstream GPU-accelerated decoder is deployed.

---

## 6. Interactive Web Deployment (Gradio)

To transform the backend logic into an accessible tool, the system was wrapped in a fully interactive web interface utilizing the **Gradio** framework. This layer serves as the **Human-Machine Interface (HMI)**, allowing for immediate visual validation of the entire pipeline without requiring code interaction.

### Interface Specifications

* **Input Layer (User Controls):**
    1.  **Visual Source:** Raw RGB image upload (Drag-and-Drop).
    2.  **Segmentation Prompt:** Text input to guide SAM3's attention (e.g., "bottles", "box") for the initial object discovery.
    3.  **Semantic Query Key:** A specific text field to test the Information Retrieval Logic (Section 5), allowing the user to query the system.

* **Output Stream (System Feedback):**
    1.  **Global Context (Object Detection):** Rendered image displaying the full scene with SAM3 segmentation masks and object bounding boxes overlayed.
    2.  **Local Detail (Barcode Localization) and Geometric Metadata:** A specialized visual output highlighting the specific Region of Interest (ROI) where the barcode was detected and rectified. Real-time textual display of the computed **Normal Vector** $(n_x, n_y, n_z)$ and the resolved **Object Label** (Identity)
    3.  **Semantic Retrieval Result:** The final text output from the Section 5 logic, displaying the data associated with the "Semantic Query Key".

---

## 7. Project Assets & Sample Outputs

The `assets/` directory contains a curated set of images that demonstrate the pipeline's artifacts at key processing stages:

* **`input`**: A sample raw RGB image used to demonstrate the expected input format.
* **`masks`**: Visualizes the binary segmentation masks generated by the SAM3 foundation model.
* **`barcodes`**: Displays the rectified Region of Interest (ROI) overlaid with the computed **Normal Vector**, illustrating the geometric orientation analysis.
* **`demo`**: A snapshot of the **Gradio Interface** in action, illustrating the user experience.

---

## 8. Models, Libraries, and Tools Used

* **Foundation Model:**
    * **SAM3 (Meta AI):** The core segmentation engine used for zero-shot object discovery and mask generation.

* **Deep Learning & Numerical Computing:**
    * **PyTorch:** The primary framework for tensor operations and loading the SAM3 architecture.
    * **Torchvision:** Used for specific geometric operations, specifically `masks_to_boxes` to convert semantic masks into bounding boxes.
    * **NumPy:** Essential for high-performance matrix manipulations and handling image arrays.

* **Computer Vision & Image Processing:**
    * **OpenCV (`cv2`):** The backbone for classical image processing tasks, including contour detection, grayscale conversion, and morphological operations used in the barcode localization pipeline.
    * **PIL (Pillow):** Used for image file handling (`Image`), drawing overlays (`ImageDraw`), and font rendering (`ImageFont`).

* **Classical Machine Learning:**
    * **Scikit-learn (`sklearn`):** Specifically the **K-Means** clustering algorithm, which was employed in "Approach 1" to attempt unsupervised grouping of barcode bar widths.

* **Interface & Visualization:**
    * **Gradio:** Used to deploy the interactive web demonstration and GUI.
    * **Matplotlib:** Utilized for plotting static figures and debugging visualization (`patches`).

* **Data Generation:**
    * **Python-barcode:** A library used to generate synthetic barcode images, crucial for creating the training dataset for the CRNN experiments.

### Technical Adaptation: Cross-Platform Compatibility
A specific modification was required to run the SAM3 architecture on a standard Windows environment without CUDA compilation dependencies.
* **The Issue:** The original SAM3 repository relies on `edt.py` (Euclidean Distance Transform), which depends on **OpenAI Triton** kernels. Triton is primarily designed for Linux and lacks native Windows support, causing runtime failures.
* **The Fix:** The dependency was replaced by a custom implementation using **SciPy**. The function `distance_transform_edt` from `scipy.ndimage` was integrated to handle the distance field calculations on the CPU, successfully bypassing the OS-specific hardware constraints.

---

## 9. System Constraints

* **Resource Dependencies (GPU):** The training of the custom CRNN decoder and the fine-tuning of foundation models are computationally intensive processes. Due to hardware constraints in the current environment, these heavy training cycles were deferred, prioritizing architectural validation over model convergence.

* **Inference Latency:** While the SAM3 architecture provides state-of-the-art segmentation, its transformer backbone incurs a high computational cost. On CPU-only infrastructure, the system operates in a batch-processing regime. Deployment on CUDA-enabled hardware is a prerequisite for achieving the millisecond-level latency required for closed-loop robotic control.

* **Verification Granularity:** The current reliance on **spatial topology** (geometric association) rather than explicit alphanumeric decoding creates a dependency on segmentation quality. In high-occlusion scenarios or cluttered scenes, this approach lacks the "ground truth" verification that OCR would provide.

* **Environmental Generalization:** Classical image processing techniques (used in Approach 1) exhibited sensitivity to photometric variations (lighting, sensor noise) and non-linear geometric distortions. This confirms the architectural decision to transition towards the proposed Deep Learning (CRNN) pipeline for production environments.

* **Geometric Ambiguity (Single-View 3D):** Estimating 3D surface normals solely from 2D PCA projections is an mathematically **ill-posed problem**. Without depth information (RGB-D), identical 2D mask shapes can correspond to divergent 3D orientations. Consequently, heuristic calibration for one object class may not generalize to others, as the system lacks the volumetric data to resolve this projection ambiguity.

---

## 10\. Potential Improvements and Future Work

* **Inference Optimization (Batching):** The current system processes prompts sequentially (iterative loop), which is computationally expensive. Future iterations should implement **batch inference** using `datapoints`, as demonstrated in the SAM3 repository examples. This would allow the model to segment all objects in a single forward pass, drastically reducing latency.
* **Perspective Rectification (Homography):** To improve normal vector estimation, the current affine transformation should be replaced by a full **Homography**.
    * *Current Assumption:* The system currently uses an **Affine Approximation**, assuming that since barcodes are small relative to the camera distance, trapezoidal perspective distortion is negligible.
    * *Improvement:* A homography would correct these trapezoidal distortions, mapping the barcode to a perfect rectangle regardless of the viewing angle.
* **Closed-Set Classification Fallback:** As an alternative to the heavy CRNN, a lightweight CNN classifier could be trained specifically on the **6 known barcode classes**. While this sacrifices generalizability (it cannot read new/unknown items), it would provide extremely fast and accurate detection for this specific, static inventory.
* **Full CRNN Training:** Completing the training of the deep learning decoder (CNN + LSTM) on a GPU cluster to achieve robust, Open-Set reading capabilities for any barcode.
* **Hardware Upgrade:** Utilization of high-VRAM GPUs is essential to accelerate SAM3 inference and enable the training of the proposed deep learning models.










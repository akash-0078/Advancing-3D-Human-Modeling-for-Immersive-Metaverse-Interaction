# Live Avatar: Mono-Cam 3D Face Rig & Emotion Pipe

This project is a high-fidelity, real-time pipeline designed to capture and animate facial expressions using only a single RGB camera. It separates identity reconstruction from emotional expression, allowing the system to generate animatable 3D face rigs and stream live data to drive digital avatars with sub-20ms latency.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Model-Wise Outputs](#model-wise-outputs)
- [Key Tech Stack](#key-tech-stack)
---

## Description

The "Live Avatar" pipeline solves the challenge of creating realistic 3D facial animations without expensive motion-capture gear. It utilizes a **Two-Stage Architecture**: 
1. **Stage 1 (Identity)**: Reconstructs a unique 3D head model (Rig) from a source image or video.
2. **Stage 2 (Expression)**: Captures live emotional "blendshapes" from a webcam and applies them to that rig in real-time. 

This system is designed for Metaverse applications, AAA gaming, and realistic telepresence where low latency and high emotional fidelity are required.

---

## Features

### Reconstruction Features:
- **3D Rig Generation**: Creates a detailed FLAME-based mesh from a single RGB source.
- **Multi-Frame Fusion**: Analyzes multiple initial frames to stabilize identity features and lighting parameters.
- **Identity Consistency**: Uses specialized loss functions to ensure the 3D model maintains the unique features of the user.

### Animation & Emotion Features:
- **EmotionLSTM**: A custom recurrent block that stabilizes frame-to-frame jitters for smooth, natural-looking animation.
- **Valence-Arousal Tracking**: Captures deep emotional states rather than just simple surface-level movements.
- **Cross-Rig Transfer**: Remaps human expressions onto stylized or non-humanoid 3D characters.
- **Low Latency Streaming**: A custom WebSocket bridge ensures data transfer happens in under 20ms.

---

## Model-Wise Outputs



### Stage 1: 3D Rig Generator
This module generates an animatable 3D face rig from a single image or video. It captures geometry, texture, and blendshape parameters for a neutral face.

<video src="assets/stage1_rig.mp4" alt="3D Rig Generator Output" width="600"/>

---

### Stage 2: Emotion Capture Module
This module analyzes live video feed to predict accurate expression and emotion coefficients in real-time.

<video src="assets/stage2_emotion.mp4" alt="Emotion Capture Output" width="600"/>

---

### Stage 3: Integrated Pipeline (Final Output)
The final integrated system where the live emotion coefficients are streamed to drive the pre-generated 3D rig with low latency.

<video src="assets/final_pipeline.mp4" alt="Full Integrated Pipeline Output" width="600"/>

---
## Key Tech Stack

Here are the core libraries and models used to build the 3D pipeline:

### Core Models:
- **FLAME**: A powerful 3D Morphable Model used as the foundational decoder for face shape and expression.
- **DECA**: Used for detailed identity reconstruction and geometry refinement.
- **EMOCA**: The backbone for capturing robust emotion and expression coefficients.

### Technical Dependencies:
- **PyTorch**: The primary deep learning framework for running the encoders and decoders.
- **MediaPipe / RetinaFace**: Used for high-speed facial landmark detection and face cropping.
- **OpenCV**: Handles real-time video processing and frame manipulation.
- **WebSockets**: Facilitates the low-latency communication between the AI models and the 3D engine.

---



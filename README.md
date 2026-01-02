# Human-Interactive 4-DOF Robotic Arm using Dual-Camera Visual Servoing

## 📌 Overview
This project presents a human-interactive 4-DOF robotic arm system capable of autonomous target reaching and social interaction through hand gesture control. Built on the LEGO Mindstorms EV3 platform, the system integrates dual-camera Image-Based Visual Servoing (IBVS) and deep learning-based gesture recognition to enable intuitive, closed-loop robotic control.

The robot can:
- Perform a pre-programmed waving motion
- Autonomously reach colored targets (blue or yellow)
- Switch behaviors using natural hand gestures

## 🎯 Objectives
- Bridge human intent and robotic action through gesture-based interaction
- Achieve closed-loop visual servoing in 3D space using dual cameras
- Implement a stable control strategy for a 4-DOF robotic arm
- Demonstrate robustness through experimental evaluation

## 🧠 Background & Motivation
Traditional industrial robots rely heavily on pre-programmed trajectories and lack adaptability in dynamic environments. This project addresses these limitations by:
- Using visual feedback to continuously correct motion
- Replacing keyboard-based input with intuitive hand gestures
- Integrating perception, control, and actuation into a unified system

## 🛠️ System Overview
The system recognizes three distinct hand gestures:
- **Wave**: Triggers a socially interactive waving motion (open-loop)
- **Down-Right**: Initiates IBVS toward a right-side colored target
- **Down-Left**: Initiates IBVS toward a left-side colored target

## 🤖 Hardware Configuration
- **Platform**: LEGO Mindstorms EV3
- **Degrees of Freedom**: 4

| Joint | Function | Motor Type |
|------|---------|------------|
| A | Base rotation | Medium Motor |
| B | Shoulder lift | Large Motor |
| C | Elbow extension | Large Motor |
| D | Wrist orientation | Medium Motor |

### 📷 Dual-Camera Setup
- **Side Camera**: Monitors depth (Z-axis)
- **Top Camera**: Monitors planar motion (X-Y axes)

## 💻 Software Architecture
The system uses a **client-server architecture over TCP/IP**:

### Robot Client (EV3)
- Runs on EV3 using `ev3dev2`
- Maps joint angles to motor positions using gear ratios
- Executes received motion commands
- Handles motor safety (brake/coast modes)

### Vision & Control Server (PC)
- Gesture recognition using **MediaPipe** and a custom **TensorFlow/Keras** model
- Dual-camera visual tracking using HSV color thresholding
- Centroid detection via Hough Circle Transform
- Online Jacobian estimation via numerical perturbation
- Control using **Damped Least Squares (DLS)** for stability

## 📐 Control Method
- Image-Based Visual Servoing (IBVS)
- Numerical Jacobian estimation by perturbing joints
- Joint velocity computed as:
- dθ = Jᵀ (J Jᵀ + λ² I)⁻¹ e
where λ is the damping factor to avoid singularities.

## 🧪 Experiments & Evaluation
The system was evaluated across multiple dimensions:

### Gesture Recognition
- Tested on Wave, Down-Right, Down-Left
- 20 trials per gesture
- Success threshold: confidence > 80%

### Reaching Accuracy
- Target placed randomly in workspace
- Convergence criteria:
  - XY error < 180 pixels
  - Z error < 100 pixels

### Repeatability
- Verified consistent return to home pose after waving
- Confirmed inverse traversal after servoing tasks

### Latency Measurement
- Measured time from gesture detection to robot motion
- Included vision processing, inference, communication, and motor inertia

## 📊 Results
- Successful trials showed monotonic decrease in XY and Z errors
- Failed trials revealed sensitivity to camera calibration and Jacobian stability
- Dual-camera setup significantly improved depth perception and convergence reliability

## ✅ Conclusion
This project demonstrates a robust semi-autonomous robotic system combining:
- Natural gesture-based interaction
- Dual-camera visual servoing
- Online Jacobian estimation
- Stable DLS control

The closed-loop design effectively compensates for mechanical inaccuracies, validating the system’s capability to operate in dynamic environments.

## 👥 Contributors
- **Yunze Liu**
- **Xindi Li**

## 📄 Reference
- LEGO EV3 Daisy Chaining Documentation  
  https://ev3-help-online.api.education.lego.com/Retail/en-us/page.html?Path=editor%2FDaisyChaining.html

## 📎 Full Report
For complete implementation details and experimental data, see:  
`Final Project Report: Human Interactive 4-DOF Robotic Arm using Dual-Camera Visual Servoing`



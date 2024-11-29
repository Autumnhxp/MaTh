![Automatic Robotic Object Detection and Retrieval](MasterThesis/doc/Automatic%20Robotic%20Object%20Detection%20and%20Retrieval.gif)
# Automatic Robotic Object Detection and Retrieval

Welcome to the **Automatic Robotic Object Detection and Retrieval** repository! This project is part of a Master's thesis, developed in collaboration with the **Human-Centered Computing and Extended Reality Lab**. The goal of this project is to implement a system for detecting and retrieving objects using robotic mechanisms, leveraging cutting-edge simulation and AI technologies.

---

## Overview

This project explores the integration of **robotic perception** and **manipulation** to perform object detection and retrieval tasks autonomously. The system is developed on NVIDIA's **Omniverse** platform, utilizing its advanced simulation and rendering capabilities for real-world scenarios.

The libraries employed include:

- **Isaac Lab**: A robust framework for robotics and AI, providing tools for simulation, motion planning, and control.
- **AnyGrasp**: A versatile library designed for generating and evaluating grasp poses for various objects, enabling reliable object manipulation.

---

## Features

- **Realistic Robotic Simulation**: Simulate object detection and retrieval tasks in a virtual environment using Omniverse.
- **Advanced Grasping Techniques**: Employ AI-driven grasping algorithms via AnyGrasp for enhanced performance.
- **Seamless Integration**: Built with Isaac Lab, combining robotic intelligence with simulation realism.
- **Scalable Design**: Adaptable to different robotic platforms and object types.

---

## Setup Instructions

### Prerequisites

1. NVIDIA **Omniverse** installed.
2. Python environment with the following dependencies:
   - `isaaclab`
   - `anygrasp`
   - Any additional Python libraries required by the Omniverse platform.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Autumnhxp/MaTh.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd MaTh
   ```
3. Creating Isaac Sim virtual environment:
   ```bash
   # create a virtual environment named isaaclab with python3.10
   python3.10 -m venv isaaclab
   # activate the virtual environment
   source isaaclab/bin/activate
   ```
4. Installing a CUDA-enabled PyTorch 2.4.0 build based on the CUDA version available on your system.
   ```bash
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
   ```
5. Installing Issac Sim:
   ```bash
   pip install --upgrade pip
   pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
   ```
   

### Running the Project
1. Execute the lift_cube_sm script to start the object detection and retrieval system in isaaclab virtual environment:
  ```bash
  (isaaclab) python /MasterThesis/lift_cube_sm.py
  ```

---

### Copyright
This project is jointly owned by the **Human-Centered Computing and Extended Reality Lab** and Master's thesis student **Xuan-Pu Autumn Hong**. All rights reserved.

---

### Acknowledgements
- **Human-Centered Computing and Extended Reality Lab** for their invaluable guidance and resources.
- **NVIDIA Omniverse** for providing a powerful simulation platform.
- Developers of **Isaac Lab** and **AnyGrasp** for their excellent libraries enabling this project.

---
> *"Shaping the future of robotics with simulation and AI."*


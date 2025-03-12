### **AnyGrasp License Setup Guide**
This folder contains license-related files for **AnyGrasp**.  
Please refer to the official **[AnyGrasp](https://github.com/graspnet/anygrasp_sdk/tree/main)** for detailed steps on **License Registration** and **Instruction of AnyGrasp Detection Demo**.

#### **Folder Structure After License Setup**
Once the license setup is completed, this folder will contain the following files:
AnyGrasp/
│── example_data/
│── license/
│   │── README.md # This guide
│   │── licenseCfg.json # License configuration file 
│   │── [YourLicense].lic # License file 
│   │── [YourName].public_key # Public key file
│   │── [YourName].signature # Signature file 
│── log/
│   │── checkpoint_detection.tar
│── Calculation_GraspPose.py
│── Calculation_GraspPose.sh
│── grasp_pose_publisher.py
│── gsnet.so # Shared library for grasp network computations
│── lib_cxx.so  # C++ shared library for low-level computations


These files are essential for AnyGrasp's license authentication.  
Ensure that all required files are present before running the software.
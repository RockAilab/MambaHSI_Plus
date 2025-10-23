🛰️ MambaHSI+
Multidirectional State Propagation for Efficient Hyperspectral Image Classification
📘 Overview

MambaHSI+ is the V2 version of our previous MambaHSI+ framework.
This version simplifies the original design while achieving better classification performance and higher computational efficiency.

The project builds upon the foundation of MambaHSI (Li et al., TGRS 2024)
, introducing multidirectional state propagation and spectral trajectory learning (STL) for more expressive and efficient hyperspectral feature modeling.

🚀 Highlights

✅ Simplified architecture (cleaner and faster than V1)

🔁 Multidirectional state propagation for enhanced spatial–spectral dependency modeling

⚡ Significantly improved classification accuracy and efficiency

🧠 Implemented using mamba-ssm==2.2.2

🧩 Training
# Train MambaHSI+
CUDA_VISIBLE_DEVICES=<gpu_id> python train_MambaHSI_Plus.py

💾 Dataset Preparation

Please refer to the Data Preparation section of
👉 MambaHSI (original repository)

🧱 Dependencies
Library	Version	Description
Python	≥3.9	Core language
PyTorch	≥1.12	Deep learning framework
mamba-ssm	2.2.2	State-space model implementation
NumPy, SciPy, scikit-learn	Latest	Data preprocessing and evaluation

📊 Citation

If you find this repository helpful in your research, please cite:

@ARTICLE{11023867, 
  author={Wang, Yunbiao and Liu, Lupeng and Xiao, Jun and Yu, Dongbo and Tao, Ye and Zhang, Wenniu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MambaHSI+: Multidirectional State Propagation for Efficient Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  pages={1-14},
  keywords={Computational modeling; Computer architecture; Transformers; Feature extraction; Trajectory; Hyperspectral imaging; Image classification; Context modeling; Accuracy; Computational efficiency; Bidirectional propagation; hyperspectral image (HSI) classification; mamba architecture; spectral trajectory learning (STL); state-space models (SSMs)},
  doi={10.1109/TGRS.2025.3576656}
}

🙏 Acknowledgment

This work is based on and inspired by the excellent prior work:

@ARTICLE{MambaHSI_TGRS24,
  author={Li, Yapeng and Luo, Yong and Zhang, Lefei and Wang, Zengmao and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification}, 
  year={2024},
  pages={1-16},
  keywords={Hyperspectral Image Classification; Mamba; State Space Models; Transformer},
  doi={10.1109/TGRS.2024.3430985}
}


We sincerely thank the authors of MambaHSI for their open-source contribution, which provided the foundation for this work.

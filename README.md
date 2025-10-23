# 🛰️ MambaHSI_Plus  
## MambaHSI+: Multidirectional State Propagation for Efficient Hyperspectral Image Classification

---

### 📘 Overview

**MambaHSI_Plus** is the **V2 version** of our previous **MambaHSI+** framework.  
This version **simplifies** the V1 architecture and achieves **better classification accuracy** and **higher computational efficiency**.

---

### 🚀 Highlights

- ✅ **Simplified architecture:** cleaner, faster, and more efficient than V1  
- 🔁 **Multidirectional state propagation:** enhances spatial–spectral dependency modeling  
- ⚡ **Improved classification accuracy and efficiency**  
- 🧠 **Implemented with:** `mamba-ssm==2.2.2`

---
### 💾 Dataset Preparation

Please refer to the **Data Preparation** section of  
👉 [**MambaHSI (original repository)**](https://github.com/li-yapeng/MambaHSI)  
for dataset downloading and preprocessing instructions.

---
### 🧱 Dependencies

| Library | Version | Description |
|----------|----------|-------------|
| Python | ≥ 3.9 | Core language |
| PyTorch | ≥ 1.12 | Deep learning framework |
| mamba-ssm | 2.2.2 | State-space model implementation |
| NumPy / SciPy / scikit-learn | Latest | Preprocessing & evaluation |

---
### 🧩 Training

To train **MambaHSI+**, run the following command:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train_MambaHSI_Plus.py
```
---
### 📊 Citation

If this work is useful in your research, please cite:

```bibtex
@ARTICLE{11023867, 
  author={Wang, Yunbiao and Liu, Lupeng and Xiao, Jun and Yu, Dongbo and Tao, Ye and Zhang, Wenniu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MambaHSI+: Multidirectional State Propagation for Efficient Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  pages={1-14},
  doi={10.1109/TGRS.2025.3576656}
}
```
---
### 🙏 Acknowledgment

This work is based on and inspired by the excellent prior research:

```bibtex
@ARTICLE{MambaHSI_TGRS24, 
  author={Li, Yapeng and Luo, Yong and Zhang, Lefei and Wang, Zengmao and Du, Bo}, 
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification}, 
  year={2024}, 
  pages={1-16},  
  doi={10.1109/TGRS.2024.3430985}
}
```
We sincerely thank the authors of MambaHSI for their open-source contribution, which provided the foundation for this work.

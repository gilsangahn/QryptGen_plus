# QryptGen+:a quantum GAN-based high-security image encryption key generator with enhanced chaotic modeling

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11128--025--04750--5-blue)](https://doi.org/10.1007/s11128-026-05081-9)
![Journal](https://img.shields.io/badge/Journal-Quantum_Information_Processing-orange)

![Submitted](https://img.shields.io/badge/Submitted-19_Jun_2025-blue)
![Accepted](https://img.shields.io/badge/Accepted-26_January_2026-green)

## 📖 About The Project

**QryptGen+** is an advanced quantum GAN-based framework designed to generate high-security image encryption keys with enhanced chaotic modeling.

Building upon our previous work (QryptGen), this project addresses structural limitations by introducing a **patch-wise generation architecture** and a **strongly entangling quantum ansatz**. Unlike traditional methods, QryptGen+ utilizes a novel loss function that explicitly maximizes entropy and promotes pixel-wise anti-correlation. This approach allows the model to effectively learn from **chaotic data distributions** (specifically the Lorenz system) and produce $28\times28$ grayscale keys with superior randomness and low spatial correlation, making them highly suitable for high-security domains such as military communication and medical image privacy.

### Key Features
* **Patch-wise Generation Architecture:** Generates larger pixel blocks ($4\times28$) to eliminate the inter-row structural correlations found in previous row-wise stacking models.
* **Strongly Entangling Ansatz:** Leverages strongly entangling layers (SELs) to enhance the quantum circuit's expressivity and capture complex chaotic dynamics.
* **Security-Oriented Loss Function:** Incorporates specific regularization terms for **entropy maximization** and **anti-correlation**, ensuring cryptographic robustness.
* **Stabilized Adversarial Training:** Adopts a balanced generator-critic update ratio (1:10) to prevent saturation drift and ensure stable convergence during training.

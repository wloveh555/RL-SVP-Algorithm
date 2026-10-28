# RL-SVP: Reinforcement Learning for Shortest Vector Problem

This repository contains the official implementation, pre-trained models, and experimental reports for the project: **"Exploration of High-Dimensional SVP Solving Based on Reinforcement Learning and Sublattice Enumeration: Systematic Evolution from 20D to 50D"**.

## 📄 Full Academic Report
👉 **[Click here to read the complete research paper (PDF)](./docs/Systematic%20Evolution%20from%2020D%20to%2050D%20in%20SVP.pdf)**

## 🚀 Key Achievements
- **SOTA Performance at 50D**: Achieved a **63.0%** breakthrough rate over the classical LLL algorithm on 50D random lattices, narrowing the gap to BKZ(45) to just 18.61%.
- **Novel Paradigm**: Abandoned traditional local Gaussian reductions (2D/3D) in favor of an **AI-driven Continuous Sliding Window + Sublattice Enumeration** strategy.
- **Support Set Coverage Law**: Discovered that the continuous window size (K) is the decisive hyperparameter. Expanding K to 40 at 50D perfectly covers the support set, triggering a qualitative performance leap.
- **Numerical Robustness**: Built-in defense mechanisms (Log-Space representation, Exact rational verification via `sympy`) to handle massive integers up to \(10^{90}\) and completely eliminate pseudo-zero vectors.

## 📂 Repository Structure
- `code/`: All training and evaluation scripts (PPO, Set Transformer, LLL/BKZ comparison).
- `models/`: Pre-trained weights for 40D and 50D agents (`.pth`).
- `demo/`: Sample lattice data (20D, 30D, 40D, 50D).
- `docs/`: Full research report and documentation.

## ⚙️ Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the comparison evaluation script for 50D models (K=35 vs K=40 vs Set Transformer):
   ```cd code
   python evaluate_50d_compare_all.py --max-lattices 20 --samples 2

## 🛠️ Technical Stack
- Algorithm: Proximal Policy Optimization (PPO), Reinforcement Learning
- Mathematics: fpylll (LLL, BKZ), sympy (Exact integer arithmetic)
- Engineering: PyTorch, Multiprocessing, CUDA

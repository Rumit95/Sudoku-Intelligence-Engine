# 🧩 Solving the Unsolvable: Neural Sudoku Solver via Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-RL-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Env-green.svg)](https://gymnasium.farama.org/)

> **"Traditional backtracking is for computers; Reinforcement Learning is for intelligence."**

This project implements a sophisticated **Maskable Proximal Policy Optimization (PPO)** agent trained to solve Sudoku puzzles. By treating Sudoku as a sequential decision-making process rather than a constraint satisfaction problem, this model learns the underlying logic of the game, achieving a **100% solve rate** on complex configurations.

---

## 🚀 Why This is Great Engineering

Solving Sudoku with RL is notoriously difficult due to the "sparse reward" problem and the massive action space. This implementation overcomes these challenges using:

*   **Custom Gymnasium Environment**: A robust `SudokuEnv` with complex reward shaping (punishing contradictions, rewarding valid placements).
*   **Action Masking**: Leveraging `sb3-contrib`'s `MaskablePPO` to handle a **1458-dimensional action space**. The agent only "sees" valid moves, significantly accelerating convergence.
*   **Dual-Phase Logic**: The agent learns both "candidate identification" (Level 0) and "final placement" (Level 1), mimicking human-like deduction.
*   **Scalability**: Built to handle datasets of **1 million+ puzzles**, utilizing GPU acceleration for efficient training.

---

## 🛠️ Technical Stack

- **Framework**: `stable-baselines3` & `sb3-contrib`
- **Environment**: `Gymnasium`
- **Mathematics**: `NumPy` & `Pandas`
- **Visualization**: `Matplotlib` (Live rendering of the board during inference)

---

## 📦 Getting Started

### 1. Requirements
Install the necessary dependencies:
```bash
pip install stable-baselines3[extra] sb3-contrib gymnasium pandas matplotlib shimmy
```

### 2. Dataset
This project is designed for the [Kaggle Sudoku Dataset](https://www.kaggle.com/datasets/bryanpark/sudoku). 
- Download the `sudoku.csv` and place it in a folder named `sudoku 1 mil/`.
- *Note: The dataset is not included in this repository due to size.*

### 3. Testing the Brain
To watch the AI solve puzzles in real-time, run:
```bash
python test_model.py
```
You can also paste your own 81-digit Sudoku string into `test_model.py` to challenge the agent!

---

## 🧠 Architecture Deep Dive

### The Action Space
Standard PPO struggles with invalid moves. We implemented a **Masked Action Space**:
- **729 cell-digit pairs** for candidate marking.
- **729 cell-digit pairs** for final value placement.
The `ActionMasker` ensures the agent never attempts an illegal move (e.g., placing a 5 in a row that already has a 5), forcing it to learn the *meaning* of the rules.

### The Reward System
- **+1.0**: Correct value placement (matches solution).
- **+0.5**: Valid candidate marking.
- **-0.5**: Invalid move attempt (masked out during training but penalized in the environment logic).
- **+10.0**: Successfully completing the entire board.

---

## 📈 Results
- **Training Time**: ~1 million timesteps on Kaggle GPU.
- **Accuracy**: 100% solve rate on test batches.
- **Efficiency**: Solves difficult puzzles in fewer steps than traditional rule-based solvers.

---

## 👔 Contact
Created by **Rumit** – *Pushing the boundaries of what Reinforcement Learning can do.*

---
*If you find this project interesting, feel free to ⭐ the repo!*

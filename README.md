# Intra-Subject Physiological Patterns of Academic Stress

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-CCE2026-red.svg)](paper/)

## Overview

This repository contains the code and analysis for the paper:

> **"Intra-Subject Physiological Patterns of Academic Stress: From Group-Level Trends to Individualized Predictions"**  
> *Submitted to CCE 2026 (23rd International Conference on Electrical Engineering, Computing Science and Automatic Control)*

We analyze the **PhysioNet Wearable Exam Stress Dataset** to investigate intra-subject relationships between physiological signals (EDA, HR, TEMP) and exam performance.

## Key Findings

| Finding | Result |
|---------|--------|
| **EDA-only accuracy** | 75.0% (LOSO validation) |
| **Optimal window** | 1 minute (vs 5 min: 68.3%) |
| **Intra-subject correlation** | r = +1.00 (S8), r = -0.77 (S1) |
| **Validation difference** | LOSO (53.3%) vs 10-fold CV (65.3%) |
| **Signal quality** | 6 participants with >20% zeros in EDA |

## Repository Structure
ESTRES/
├── paper/ # LaTeX source and figures for CCE 2026
│ ├── figures/ # All figures (28 files)
│ ├── tables/ # LaTeX tables
│ └── scripts/ # Paper-related scripts
├── scripts/ # Main analysis scripts
│ └── analysis/ # Detailed analysis scripts
├── results/ # Generated results
│ ├── figures/ # Output figures
│ └── tables/ # Output tables (CSV)
├── analysis/ # Exploratory analysis (legacy)
├── notebooks/ # Jupyter notebooks
├── data/ # Dataset (not included, see instructions)
├── requirements.txt # Python dependencies
├── run_all.py # Main pipeline
└── README.md

text

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ErickToque/ESTRES.git
cd ESTRES
2. Create virtual environment
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Download dataset
Download the PhysioNet Wearable Exam Stress Dataset from:
https://physionet.org/content/wearable-exam-stress/1.0/

Place the contents in data/ folder.

5. Run analysis
bash
python run_all.py
 Results
Generated figures and tables will be saved in results/figures/ and results/tables/.

Main results include:
Intra-subject correlation tables

Window optimization plots

Ablation study results (EDA, HRV, TEMP, ACC)

Validation strategy comparison (LOSO vs 10-fold CV)

 Citation
bibtex
@inproceedings{toque2026intra,
  title={Intra-Subject Physiological Patterns of Academic Stress: From Group-Level Trends to Individualized Predictions},
  author={Toque, Erick},
  booktitle={2026 23rd International Conference on Electrical Engineering, Computing Science and Automatic Control (CCE)},
  year={2026}
}
 License
This project is licensed under the MIT License - see the LICENSE file for details.

 Acknowledgments
PhysioNet for the Wearable Exam Stress Dataset

Pontificia Universidad Católica del Perú for academic support

CCE 2026 for the publication opportunity

 Contact
Erick Toque - estoque@pucp.edu.pe

Project Link: https://github.com/ErickToque/ESTRES

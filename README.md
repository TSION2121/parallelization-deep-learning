
# Parallelization of Deep Learning Models

[![Report](https://img.shields.io/badge/Documentation-Technical%20Report-blue)](report/Take_Home.pdf)  

## path to the report:  report/Take_Home.pdf

This repository contains the implementation, evaluation, and performance analysis of parallelization strategies for training Deep Learning models. The project compares a **Serial Baseline** against **Data Parallel** and **Hybrid/Distributed** architectures using Convolutional Neural Networks (CNN).

## 📂 Repository Structure

```text
├── notebooks/           
│   ├── Hybrid.ipynb             # Hybrid/DDP implementation with CIFAR-10
│   └── parallelization_cnn.ipynb # Serial vs. DataParallel with MNIST
├── report/              
│   ├── Take_Home.pdf            # Comprehensive 5-page Technical Report
│   └── technical_report.md      # Markdown version of the report
├── src/                 
│   ├── serial_trainer.py        # Baseline training script
│   └── parallel_trainer.py      # Shared-memory parallel script
├── results/             # Visualization plots (Loss, Accuracy, Speedup)
└── README.md            # Reproduction instructions

```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/TSION2121/parallelization-deep-learning.git
cd parallelization-deep-learning
pip install -r requirements.txt

```

### 2. Running the Experiments

* **Serial Implementation:** Run the baseline on a single device.
```bash
python src/serial_trainer.py

```


* **Data Parallel Implementation:** Utilize `torch.nn.DataParallel` for shared-memory parallelism.
```bash
python src/parallel_trainer.py

```


* **Hybrid/Distributed (DDP):** For advanced scaling and CIFAR-10 analysis, execute the `notebooks/Hybrid.ipynb` in a GPU-enabled environment (e.g., Google Colab or AWS).

## 📊 Performance Comparison

We analyzed the training efficiency across two datasets (MNIST and CIFAR-10). The following table summarizes the speedup achieved through parallelization:

| Strategy | Architecture | Parallel Type | Avg. Speedup |
| --- | --- | --- | --- |
| **Serial** | Single GPU/CPU | None | 1.00x (Baseline) |
| **Data Parallel** | Multi-GPU | Shared-Memory | ~1.85x |
| **Hybrid (DDP)** | Distributed | Hybrid/All-Reduce | ~2.15x |

### Key Results:

* **Accuracy Verification:** Both parallel and hybrid models reached the same accuracy as the serial baseline, proving that the parameter synchronization logic is correct.
* **Scalability:** Hybrid parallelism (DDP) outperformed standard Data Parallelism by reducing the communication bottleneck on the master GPU.

## 🛠 Challenges & Optimizations

* **Communication Overhead:** Addressed by implementing Distributed Data Parallel (DDP) to use All-Reduce instead of parameter gathering.
* **Data Bottlenecks:** Optimized the data pipeline using `num_workers` in the DataLoader to ensure the GPU is never idling while waiting for CPU pre-processing.

## 📝 Assignment Deliverables Checklist

* [x] **Serial Source Code:** Found in `src/serial_trainer.py`.
* [x] **Parallel Source Code:** Found in `src/parallel_trainer.py` and `notebooks/`.
* [x] **Technical Report:** 5-page analysis located in `report/Take_Home.pdf`.
* [x] **Visualizations:** Loss curves and performance tables included in notebooks and results.

---





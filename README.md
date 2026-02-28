# PULMO-AI 🫁

**An integrated acoustic-microwave fusion platform for non-invasive, accessible lung cancer screening**

PULMO-AI is a research-driven biomedical engineering project that combines microwave imaging, acoustic analysis, and deep learning to create an affordable, portable lung cancer screening system. By fusing structural microwave data with functional acoustic signatures, PULMO-AI aims to democratize early lung cancer detection—especially in underserved and remote communities where traditional diagnostic infrastructure is unavailable.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Components](#hardware-components)
- [Software Components](#software-components)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Microwave Imaging Subsystem](#microwave-imaging-subsystem)
- [Acoustic Analysis Subsystem](#acoustic-analysis-subsystem)
- [Multi-Modal Fusion](#multi-modal-fusion)
- [Results](#results)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citations](#citations)

---

## 🔭 Overview

Lung cancer remains the leading cause of cancer death worldwide, largely due to late-stage diagnosis. Current screening methods like low-dose CT scans are effective but face significant barriers:

- **💰 Cost:** CT scanners cost $100,000+ and require specialized facilities
- **☢️ Radiation:** Cumulative radiation exposure limits repeated screening
- **🌍 Accessibility:** Rural and remote communities lack access to screening infrastructure
- **👨‍⚕️ Expertise:** Shortage of radiologists to interpret results, especially in low-income regions

PULMO-AI addresses these challenges through a novel **dual-modality fusion approach** combining:

1. **Microwave imaging** — Safe, non-ionizing structural sensing of lung tissue using ultra-wideband antennas
2. **Acoustic analysis** — Functional assessment of lung health through digital auscultation and deep learning
3. **Multi-modal fusion** — Combined analysis that outperforms either modality alone

The entire system is built on low-cost, off-the-shelf components and is completely **open-source**, enabling communities to build, adapt, and deploy their own screening solutions.

---

## ✨ Features

### 🔬 Microwave Imaging
- **4-antenna switched array** for multi-angle transmission measurements
- **2-3 GHz frequency range** optimized for lung tissue penetration
- **S21 parameter analysis** for detecting dielectric property variations
- **Tumor detection threshold:** ~4.9 dB signal drop for 2cm simulated masses
- **Spatial sensitivity:** Signal variation with tumor position enables localization

### 🎧 Acoustic Analysis
- **YAMNet-based feature extraction** (1,024-dimensional embeddings)
- **5-class classification:** COPD, Asthma, Pneumonia, Healthy, Bronchial
- **Digital stethoscope integration** with dual-microphone setup
- **Real-time audio processing** capability

### 🤖 Multi-Modal Fusion
- **Feature-level concatenation** of microwave and acoustic embeddings
- **XGBoost classifier** for unified diagnostic prediction
- **Confidence scoring** for clinical decision support
- **Explainability hooks** for future Grad-CAM integration

### 🌍 Democratization
- **Total system cost:** < $500 (fraction of traditional alternatives)
- **Open-source hardware designs** (3D-printable enclosures)
- **Complete software stack** available on GitHub
- **Educational course** (in development) for community replication

---

## 🏗️ System Architecture

---

## 🧰 Hardware Components

### Core Components
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| Raspberry Pi 4 (4GB) | 1 | Main processor & control | $90.99 |
| NanoVNA-F V2 | 1 | Vector network analyzer for S21 measurement | $89.99 |
| Mini-Circuits ZFSWA-2-46 RF Switches | 2 | SPDT switches for antenna multiplexing | $55.99 (used) |
| UWB Vivaldi Antennas | 4-6 | Microwave transmission/reception | $53.94 (6pk) |
| SMA Cables (Bingfu) | Multiple | RF signal routing | $45.99 |
| 2N2222 NPN Transistors | 2 | Switch control interface | Included in kit |
| 1kΩ Resistors | 2 | Current limiting for transistors | Included in kit |
| Breadboard | 1 | Prototyping circuit assembly | Included in kit |
| Jumper Wires & Alligator Clips | Various | Connections to RF switches | Included in kit |

### Acoustic Components
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| BOYA BY-M1S Lavalier Microphones | 2 | Stereo lung sound capture | $49.30 |
| Primacare Stethoscope (modified) | 2 | Professional acoustic coupling | $24.90 |
| UGREEN USB-C to 3.5mm Adapter | 2 | Microphone connection to Pi | $31.98 |

### Phantom Materials
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| Agar Agar Powder | 1 | Tissue-mimicking gel base | $14.95 |
| Polyethylene Foam Padding | 1 | Low-signal-loss enclosure | $25.89 |
| Clear Silicone Caulk | 1 | Low-loss adhesive | $32.90 |
| 7-inch HDMI Touch Display | 1 | Interactive GUI (optional) | $69.99 |

**Total System Cost:** ~$450-500 (significantly less than traditional medical imaging)

---

## 💻 Software Components

### Core Scripts

| Script | Description |
|--------|-------------|
| `switch_controller.py` | Handles digital control of Tx and Rx RF switches via GPIO |
| `vna_interface.py` | Interfaces with NanoVNA to acquire S21 transmission data |
| `array_scanner.py` | Iterates through antenna combinations and records measurements |
| `calibration.py` | Applies baseline subtraction and normalization |
| `feature_extraction.py` | Converts raw S21 traces into ML-ready features |
| `audio_processor.py` | Captures and processes lung sounds via YAMNet |
| `fusion_classifier.py` | Combines microwave and acoustic features for unified prediction |
| `main_cli.py` | Command-line interface for automated scanning |

### Dependencies

```bash
# Core dependencies
pip install numpy pandas scikit-learn matplotlib
pip install tensorflow  # for YAMNet
pip install pyserial    # for VNA communication
pip install RPi.GPIO    # for switch control (on Pi)

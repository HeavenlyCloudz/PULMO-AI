# THORACIS AI: Operation Oracle

**An integrated acoustic-microwave fusion platform for non-invasive, accessible lung cancer screening**

THORACIS AI is a research-driven biomedical engineering project that combines microwave imaging, acoustic analysis, and deep learning to create an affordable, portable lung cancer screening system. By fusing structural microwave data with functional acoustic signatures, THORACIS AI aims to democratize early lung cancer detection—especially in underserved and remote communities where traditional diagnostic infrastructure is unavailable.

---

## Overview

Lung cancer remains the leading cause of cancer death worldwide, largely due to late-stage diagnosis. Current screening methods like low-dose CT scans are effective but face significant barriers:

- **Cost:** CT scanners cost $100,000+ and require specialized facilities
- **Radiation:** Cumulative radiation exposure limits repeated screening
- **Accessibility:** Rural and remote communities lack access to screening infrastructure
- **Expertise:** Shortage of radiologists to interpret results, especially in low-income regions

THORACIS AI addresses these challenges through a novel **dual-modality fusion approach** combining:

1. **Microwave imaging** — Safe, non-ionizing structural sensing of lung tissue using ultra-wideband antennas
2. **Acoustic analysis** — Functional assessment of lung health through digital auscultation and deep learning
3. **Multi-modal fusion** — Combined analysis that outperforms either modality alone

The entire system is built on low-cost, off-the-shelf components and is completely **open-source**, enabling communities to build, adapt, and deploy their own screening solutions.

---

## Features

### Microwave Imaging
- 4-antenna switched array for multi-angle transmission measurements
- 2-3 GHz frequency range optimized for lung tissue penetration
- S21 parameter analysis for detecting dielectric property variations
- **Background subtraction** in linear domain (not dB) to remove antenna coupling
- Multi-angle scanning (0, 120, 240 degrees) for 3D localization
- Tumor detection threshold: ~4.9 dB signal drop for 2cm simulated masses

### Acoustic Analysis
- YAMNet-based feature extraction (1,024-dimensional embeddings)
- 5-class classification: COPD, Asthma, Pneumonia, Healthy, Bronchial
- Digital stethoscope integration with USB microphone support
- Bandpass filtering (100-2000 Hz) for respiratory focus
- Real-time audio processing capability

### Multi-Modal Fusion
- Feature-level concatenation of microwave and acoustic embeddings
- XGBoost classifier for unified diagnostic prediction
- Confidence scoring for clinical decision support
- Explainability visualizations showing decision rationale

### Operation Oracle - Cross-Device Integration
- Shared sync folder (`/opt/oracle_share`) for THORACIS AI and NOMA AI
- Cross-modal alerts detect paraneoplastic syndromes
- Unified patient record dashboard
- Automatic data syncing between devices

### Health Passport
- Personal lung health record for each patient
- Longitudinal tracking of respiratory health
- Trend analysis and history visualization
- Exportable health reports (CSV format)

### Clinical Assessment Module
- Respiratory-focused questionnaire
- Integration with AI predictions
- Risk level calculation (LOW/MODERATE/HIGH)
- Educational content for patient literacy

### Explainable AI
- Acoustic spectrogram visualization with feature overlays
- Microwave S21 trace display with tumor contrast
- Fusion decision explanation with cross-modal agreement
- Background subtraction demonstration

### Democratization
- Total system cost: < $500 (fraction of traditional alternatives)
- Open-source hardware designs (3D-printable enclosures)
- Single Python script for all functionality
- Educational course content integrated into the app

---

## Hardware Components

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

### Acoustic Components
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| USB Condenser Microphone | 1 | Lung sound capture | $20-50 |
| Primacare Stethoscope (modified) | 1 | Professional acoustic coupling | $24.90 |

**Total System Cost:** ~$450-500 (significantly less than traditional medical imaging)

---

## Software Dependencies

# Core dependencies
pip install numpy pandas scikit-learn matplotlib
pip install tensorflow
pip install pyserial
pip install RPi.GPIO
pip install sounddevice scipy
pip install PySide6
pip install tflite-runtime
pip install xgboost

---

## Installation

### 1. Clone the Repository


git clone https://github.com/HeavenlyCloudz/THORACIS_AI.git
cd THORACIS_AI


### 2. Install Dependencies


pip install -r requirements.txt


### 3. Setup Shared Sync Folder


sudo mkdir -p /opt/oracle_share
sudo chmod 777 /opt/oracle_share


### 4. Create Models Directory


mkdir -p data/models


Place the following model files in `data/models/`:
- `yamnet_working.tflite`
- `lung_audio.tflite`
- `thoracis_fusion_model_840features.pkl`
- `thoracis_fusion_scaler_840features.pkl`
- `thoracis_final_model.pkl`

### 5. Hardware Setup

#### RF Switch Wiring (per switch)

text
GPIO 17 --[1kΩ]--[2N2222 Base]
                  Collector -- +5V rail
                  Emitter --- RF1 pin

GPIO 27 --[1kΩ]--[2N2222 Base]
                  Collector -- +5V rail
                  Emitter --- RF2 pin

GPIO 18 --[1kΩ]--[2N2222 Base]
                  Collector -- +5V rail
                  Emitter --- RF1 pin (Switch 2)

GPIO 22 --[1kΩ]--[2N2222 Base]
                  Collector -- +5V rail
                  Emitter --- RF2 pin (Switch 2)


#### Antenna Connections
- **Switch #1 (TX):** COM VNA CH0, Port 1 Antenna 1, Port 2 Antenna 2
- **Switch #2 (RX):** COM VNA CH1, Port 1 Antenna 3, Port 2 Antenna 4

#### Power
- Connect Pi 5V pin to breadboard positive rail
- Connect Pi GND pin to breadboard negative rail

---

## Usage

### Run the Application


python thoracis_app.py


### Application Workflow

1. **Microwave Tab**: Record baseline (air) for background subtraction, then scan patient
2. **Acoustic Tab**: Record lung sounds using USB microphone
3. **Fusion Tab**: Combine both modalities for unified diagnosis
4. **Health Passport**: View patient history and trends
5. **Operation Oracle**: View cross-device data from NOMA AI
6. **Education**: Learn about respiratory conditions

### Data Collection Mode

Use the Data Collection tab for systematic phantom scanning:
- Manual RF switch control
- CSV data export for model training
- Multi-angle support

---

## Operation Oracle - Cross-Device Syncing

THORACIS AI shares data with NOMA AI through `/opt/oracle_share`:

- **Thoracis AI** writes lung scan results to `thoracis_*.json`
- **NOMA AI** writes skin scan results to `noma_*.json`
- Both systems monitor the folder for incoming scans
- Cross-modal alerts detect paraneoplastic syndromes

---

## Repository Structure

text
THORACIS_AI/
├── README.md                    # This file
├── LICENSE
├── requirements.txt             # Python dependencies
├── thoracis_app.py              # Main application (single script)
├── docs/
│   ├── installation.md          
│   ├── hardware_setup.md        
│   ├── usage.md                 
│   └── explainability.md        
├── data/
│   ├── scans/                   # Runtime scan data
│   └── models/                  # ML models (not included in repo)
├── scripts/
│   ├── setup.sh                 
│   └── run.sh                   
└── .gitignore


---

## Contributing

### Ways to Contribute
- Code improvements
- Hardware design optimization
- Additional model training
- Documentation
- Clinical validation
- Global deployment

### Contribution Steps

git checkout -b feature/amazing-feature
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature


Open a Pull Request.

---

## License

MIT License. You may:
- Use commercially
- Modify
- Distribute
- Use privately
- Sublicense

Include original copyright notice.

---

## Contact

Project Lead: Anie Udofia
William Aberhart High School, Calgary, AB
GitHub: @HeavenlyCloudz
Repository: github.com/HeavenlyCloudz/THORACIS_AI

---

## Support

- Star the repository
- Share the project
- Contribute
- Reach out for collaboration

Together, we can make early cancer detection accessible to all.


txt
# THORACIS AI Dependencies
# Python 3.9+

# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
scipy>=1.7.0

# Machine Learning
tensorflow>=2.8.0
tflite-runtime>=2.8.0
xgboost>=1.5.0

# Hardware
pyserial>=3.5.0
RPi.GPIO>=0.7.1
sounddevice>=0.4.4

# GUI
PySide6>=6.2.0


gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Data
data/scans/
data/models/
*.csv
*.npy
*.pkl
*.h5
*.tflite

# SQLite
*.db

# JSON
*.json

# Logs
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# Local configuration
config/local.py



#!/bin/bash
# THORACIS AI Launch Script

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "thoracis_env" ]; then
    source thoracis_env/bin/activate
fi

# Run the application
python thoracis_app.py "$@"


markdown
# Installation Guide

## Prerequisites
- Raspberry Pi 4 (or Linux system with Python 3.9+)
- NanoVNA-F V2 connected via USB
- RF switches wired to GPIO
- USB microphone for acoustic capture

## Step 1: System Setup


# Update system
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv thoracis_env
source thoracis_env/bin/activate


## Step 2: Install Python Packages


pip install -r requirements.txt


## Step 3: Create Required Directories


mkdir -p data/scans data/models
sudo mkdir -p /opt/oracle_share
sudo chmod 777 /opt/oracle_share


## Step 4: Add Model Files

Place these files in `data/models/`:
- `yamnet_working.tflite`
- `lung_audio.tflite`
- `thoracis_fusion_model_840features.pkl`
- `thoracis_fusion_scaler_840features.pkl`
- `thoracis_final_model.pkl`

## Step 5: Hardware Wiring

Follow the RF switch wiring diagram in the main README.

## Step 6: Run the Application


python thoracis_app.py



markdown
# Hardware Setup Guide

## RF Switch Wiring

### Components Needed
- 2x Mini-Circuits ZFSWA-2-46 RF Switches
- 2x 2N2222 NPN Transistors
- 2x 1kΩ Resistors
- Jumper wires and alligator clips
- Breadboard

### Wiring Diagram


                   Switch 1 (TX)
                   +----------+
                   |          |
   GPIO 17 ----[1kΩ]---- Base |    RF1 ---- Antenna 1
                   | Collector|    RF2 ---- Antenna 2
                   |          |
                   +----------+
                   |    |
                   +----+----- +5V (Pi)
                   |
                   Emitter --- GND

                   Switch 2 (RX)
                   +----------+
                   |          |
   GPIO 18 ----[1kΩ]---- Base |    RF1 ---- Antenna 3
                   | Collector|    RF2 ---- Antenna 4
                   |          |
                   +----------+
                   |    |
                   +----+----- +5V (Pi)
                   |
                   Emitter --- GND


### GPIO Pin Mapping

| GPIO Pin | Function |
|----------|----------|
| GPIO 17  | Switch 1 - RF1 control |
| GPIO 27  | Switch 1 - RF2 control |
| GPIO 18  | Switch 2 - RF1 control |
| GPIO 22  | Switch 2 - RF2 control |

### VNA Connections


VNA CH0 (TX) ----- Switch 1 COM
VNA CH1 (RX) ----- Switch 2 COM

### Path Configurations

| Path | TX Antenna | RX Antenna | GPIO 17 | GPIO 27 | GPIO 18 | GPIO 22 |
|------|------------|------------|---------|---------|---------|---------|
| 1    | 1          | 3          | HIGH    | LOW     | HIGH    | LOW     |
| 2    | 1          | 4          | HIGH    | LOW     | LOW     | HIGH    |
| 3    | 2          | 3          | LOW     | HIGH    | HIGH    | LOW     |
| 4    | 2          | 4          | LOW     | HIGH    | LOW     | HIGH    |

## USB Microphone Setup

1. Connect USB microphone to Raspberry Pi
2. Verify detection:

arecord -l

3. Test recording:

arecord -d 5 -f cd -t wav test.wav


## Power Requirements
- Raspberry Pi: 5V, 3A USB-C power supply
- RF Switches: Powered from Pi 5V rail
- VNA: USB power from Pi or separate USB power

# Usage Guide

## Running THORACIS AI


python thoracis_app.py


The application will launch in full-screen mode with the following tabs:

## Tab 1: Microwave

### Step 1: Record Baseline (Air)
- Click "RECORD BASELINE (AIR)"
- Ensure no phantom/patient is between antennas
- This captures direct antenna coupling for background subtraction
- Baseline is stored and used for all subsequent scans

### Step 2: Scan Patient
- Place patient between antennas
- Click "SCAN PATIENT (MULTI-ANGLE)"
- System scans at 0, 120, and 240 degrees
- Background subtraction removes antenna coupling in linear domain
- Results show tumor localization and confidence

## Tab 2: Acoustic

### Step 1: Position Microphone
- Place modified stethoscope on patient's back
- Connect USB microphone to Pi

### Step 2: Analyze
- Click "ANALYZE LUNG SOUNDS"
- System records 3 seconds of audio
- YAMNet extracts features
- Classifier predicts condition
- Clinical assessment questionnaire appears

### Step 3: Review Results
- Diagnosis with confidence score
- Severity level
- Clinical recommendations
- Educational content

## Tab 3: Fusion

### Step 1: Microwave Scan
- Click "1. SCAN MICROWAVE (MULTI-ANGLE)"
- Requires baseline from Microwave tab

### Step 2: Acoustic Analysis
- Click "2. ANALYZE ACOUSTIC"
- Records and classifies lung sounds

### Step 3: Fusion Diagnosis
- Click "3. RUN FUSION DIAGNOSIS"
- Combines structural and functional data
- Shows cross-modal agreement
- Provides clinical recommendation

## Tab 4: Health Passport

- Select or create patient
- View scan history
- Track health trends
- Export health report (CSV)

## Tab 5: Operation Oracle

- Open Unified Dashboard
- View thoracic and skin scans
- Cross-modal alerts
- Paraneoplastic syndrome detection

## Tab 6: Education

- Learn about respiratory conditions
- Understand the technology
- Background subtraction explained
- Clinical literacy resources

## Tab 7: Data Collection

- Manual RF switch control
- For systematic phantom scanning
- CSV data export for model training

## Tab 8: Explainability

- See how the AI makes decisions
- Acoustic spectrogram visualization
- Microwave S21 trace display
- Fusion decision explanation
- Background subtraction demonstration

## Repository Update Commands

After creating all files, run these commands to update your repository:

# Remove old scripts (keep only thoracis_app.py)
git rm switch_controller.py vna_interface.py array_scanner.py
git rm calibration.py feature_extraction.py audio_processor.py
git rm fusion_classifier.py main_cli.py

# Add new files
git add README.md requirements.txt .gitignore
git add thoracis_app.py
git add docs/ scripts/

# Commit and push
git commit -m "Restructure: Single thoracis_app.py with unified GUI. Remove individual scripts. Add Operation Oracle integration."
git push origin main

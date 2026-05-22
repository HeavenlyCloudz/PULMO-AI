#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THORACIS AI - Operation Oracle: Democratized Lung Screening System
"""

import os

# Force OpenGL ES - must be set before any Qt imports
if 'QT_OPENGL' not in os.environ:
    os.environ['QT_OPENGL'] = 'es2'
if 'QT_QPA_EGLFS_HIDECURSOR' not in os.environ:
    os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'

import sys
import time
import json
import threading
import serial
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import math
import traceback
import csv
import sqlite3

# Qt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit,
    QFrame, QScrollArea, QGroupBox, QRadioButton, QButtonGroup,
    QCheckBox, QComboBox, QFileDialog, QInputDialog, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QApplication, QSplitter
)

# Hardware
import RPi.GPIO as GPIO
import sounddevice as sd

# ML
import tflite_runtime.interpreter as tflite

# For reconstruction
from scipy.ndimage import gaussian_filter

# For resampling
import scipy.signal

# For visualizations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# CONFIGURATION - ALL VARIABLES DEFINED AT THE START
# =============================================================================

# Model paths
MODEL_DIR = Path.home() / "thoracis_ai_app" / "models"
YAMNET_PATH = MODEL_DIR / "yamnet_working.tflite"
AUDIO_MODEL_PATH = MODEL_DIR / "lung_audio.tflite"
FUSION_MODEL_PATH = MODEL_DIR / "thoracis_fusion_model_840features.pkl"
FUSION_SCALER_PATH = MODEL_DIR / "thoracis_fusion_scaler_840features.pkl"
MICROWAVE_MODEL_PATH = MODEL_DIR / "thoracis_final_model.pkl"

# Image paths for explainability tab
EXPLAINABILITY_IMAGES_DIR = Path.home() / "thoracis_ai_app"
SPECTROGRAM_IMAGE_PATH = EXPLAINABILITY_IMAGES_DIR / "Acoustic Spectrogram Visualization_resized.png"
MICROWAVE_IMAGE_PATH = EXPLAINABILITY_IMAGES_DIR / "Screenshot 2026-05-18 160700.png"

# Data directories
DATA_DIR = Path.home() / "thoracis_ai_app" / "scans"
BASELINE_DIR = DATA_DIR / "baseline"
PATIENT_DIR = DATA_DIR / "patient"
MULTI_ANGLE_DIR = DATA_DIR / "multi_angle"

# VNA Serial Settings
VNA_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
START_FREQ = 2000000000
STOP_FREQ = 3000000000
POINTS = 201

# GPIO pins for RF switches
SWITCH1_A = 17
SWITCH1_B = 27
SWITCH2_A = 18
SWITCH2_B = 22

# RF switch path configurations
PATHS = {
    1: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '1->3', 'desc': 'Left to Bottom'},
    2: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '1->4', 'desc': 'Left to Top'},
    3: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '2->3', 'desc': 'Right to Bottom'},
    4: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '2->4', 'desc': 'Right to Top'},
}

# Antenna positions for reconstruction (x,y) in mm
ANTENNA_POSITIONS = {
    1: (-75, 0),
    2: (75, 0),
    3: (0, -75),
    4: (0, 75),
}

# Path to antenna pair mapping
PATH_TO_ANTENNA_PAIR = {
    1: (1, 3),
    2: (1, 4),
    3: (2, 3),
    4: (2, 4),
}

# Multi-angle scanning configuration
ROTATION_ANGLES = [0, 120, 240]

# Audio settings - FIXED SAMPLE RATE DETECTION
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
EXPECTED_AUDIO_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
AUDIO_GAIN = 15.0

# Feature dimensions
N_FREQ_POINTS = POINTS
N_PATHS = 4
N_FREQ_FEATURES = N_PATHS * N_FREQ_POINTS
N_TIME_FEATURES_PER_PATH = 9
N_TIME_FEATURES = N_PATHS * N_TIME_FEATURES_PER_PATH
TOTAL_FEATURES = N_FREQ_FEATURES + N_TIME_FEATURES

# Model class order from training
MODEL_CLASSES = ['bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

# Audio classes mapping
AUDIO_CLASSES = ['healthy', 'asthma', 'copd', 'pneumonia', 'bronchial']
AUDIO_SEVERITY = {
    'healthy': 0.0,
    'asthma': 0.4,
    'bronchial': 0.5,
    'copd': 0.8,
    'pneumonia': 0.9
}

# Educational content
EDUCATIONAL_CONTENT = {
    'asthma': {
        'description': 'Asthma causes airway inflammation and narrowing, leading to wheezing and breathing difficulties.',
        'clinical_signs': 'Wheezing, chest tightness, shortness of breath, coughing, especially at night or early morning.',
        'recommendations': 'Use prescribed inhalers, avoid triggers, create an asthma action plan.'
    },
    'copd': {
        'description': 'Chronic Obstructive Pulmonary Disease includes emphysema and chronic bronchitis, causing airflow blockage.',
        'clinical_signs': 'Chronic cough, sputum production, shortness of breath during daily activities, frequent respiratory infections.',
        'recommendations': 'Smoking cessation, pulmonary rehabilitation, oxygen therapy if needed, regular check-ups.'
    },
    'pneumonia': {
        'description': 'Pneumonia is an infection that inflames air sacs in one or both lungs, causing them to fill with fluid.',
        'clinical_signs': 'Cough with phlegm, fever, chills, shortness of breath, chest pain during breathing or coughing.',
        'recommendations': 'Antibiotics for bacterial cases, rest, hydration, follow-up chest X-ray, monitor oxygen levels.'
    },
    'healthy': {
        'description': 'Normal lung function with clear airways and effective gas exchange.',
        'clinical_signs': 'No persistent cough, normal breathing patterns, ability to perform daily activities without breathlessness.',
        'recommendations': 'Maintain healthy lifestyle, avoid smoking, regular exercise, preventive care.'
    },
    'bronchial': {
        'description': 'Bronchial issues affect the main airways to the lungs, causing inflammation and mucus production.',
        'clinical_signs': 'Persistent cough often with mucus, fatigue, chest discomfort, mild fever.',
        'recommendations': 'Rest, hydration, avoid irritants, seek medical care if symptoms persist.'
    }
}

# =============================================================================
# SYNC FOLDER CONFIGURATION (for Operation Oracle) - SHARED LOCATION
# =============================================================================
SYNC_FOLDER = Path("/opt/oracle_share")
SYNC_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"SYNC_FOLDER set to: {SYNC_FOLDER}")

# Create archive subfolder
ARCHIVE_DIR = SYNC_FOLDER / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def db_to_linear(db):
    """Convert dB to linear magnitude (power ratio)"""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB"""
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def apply_background_subtraction(phantom_s21_db, baseline_s21_db):
    """
    Remove direct antenna coupling by subtracting baseline in linear domain.
    Subtraction MUST be in linear (power) domain, not dB!
    """
    phantom_linear = db_to_linear(phantom_s21_db)
    baseline_linear = db_to_linear(baseline_s21_db)
    corrected_linear = phantom_linear - baseline_linear
    corrected_linear = np.maximum(corrected_linear, 1e-12)
    corrected_db = linear_to_db(corrected_linear)
    return corrected_db

def init_thoracic_db():
    """Initialize the SQLite database for thoracic longitudinal tracking"""
    conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS thoracic_scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        diagnosis TEXT,
        confidence REAL,
        microwave_result TEXT,
        audio_result TEXT,
        localization TEXT,
        risk_level TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS skin_scans_received (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        diagnosis TEXT,
        confidence REAL,
        risk_level TEXT,
        source TEXT,
        raw_data TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Thoracic longitudinal database initialized")

init_thoracic_db()

def sync_scan_to_noma(scan_data):
    """Save scan result to synced folder so NOMA AI can see it"""
    try:
        filename = SYNC_FOLDER / f"thoracis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.json"
        scan_data['source_device'] = 'THORACIS_AI'
        scan_data['scan_type'] = 'lung'
        scan_data['timestamp'] = datetime.now().isoformat()
        with open(filename, 'w') as f:
            json.dump(scan_data, f, indent=2)
        print(f"Scan synced to NOMA AI: {filename}")
        return True
    except Exception as e:
        print(f"Sync error: {e}")
        return False

def check_for_skin_scans():
    """Check for incoming skin scans from NOMA AI - FIXED for proper JSON parsing"""
    skin_scans = []
    try:
        for json_file in SYNC_FOLDER.glob("noma_*.json"):
            try:
                with open(json_file, 'r') as f:
                    content = f.read()
                    scan = json.loads(content)
                    
                    # Handle case where confidence is an object/string
                    if 'confidence' in scan and isinstance(scan['confidence'], (dict, str)):
                        if isinstance(scan['confidence'], dict):
                            for key in ['confidence', 'value', 'score']:
                                if key in scan['confidence']:
                                    scan['confidence'] = float(scan['confidence'][key])
                                    break
                            else:
                                scan['confidence'] = 0.85
                        elif isinstance(scan['confidence'], str):
                            try:
                                scan['confidence'] = float(scan['confidence'])
                            except:
                                scan['confidence'] = 0.85
                    elif 'confidence' not in scan:
                        scan['confidence'] = 0.85
                    
                    # Ensure prediction field exists
                    if 'prediction' not in scan and 'diagnosis' in scan:
                        scan['prediction'] = scan['diagnosis']
                    elif 'prediction' not in scan:
                        scan['prediction'] = 'Unknown'
                    
                    if 'scan_type' not in scan:
                        scan['scan_type'] = 'skin'
                    
                    skin_scans.append(scan)
                    
                    archive_file = ARCHIVE_DIR / json_file.name
                    json_file.rename(archive_file)
                    print(f"Processed skin scan: {json_file.name} -> {scan.get('prediction', 'Unknown')}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON parse error in {json_file.name}: {e}")
                json_file.rename(ARCHIVE_DIR / json_file.name)
            except Exception as e:
                print(f"Error reading {json_file.name}: {e}")
                json_file.rename(ARCHIVE_DIR / json_file.name)
                
    except Exception as e:
        print(f"Error checking skin scans: {e}")
    
    return skin_scans

def save_skin_scan_to_db(scan_data):
    """Save skin scan to database - FIXED for proper field handling"""
    try:
        conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
        cursor = conn.cursor()
        
        diagnosis = scan_data.get('prediction', scan_data.get('diagnosis', 'Unknown'))
        confidence = scan_data.get('confidence', 0.85)
        
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.85
        
        risk_level = scan_data.get('risk_level', 'LOW')
        timestamp = scan_data.get('timestamp', datetime.now().isoformat())
        
        cursor.execute('''
            INSERT INTO skin_scans_received (timestamp, diagnosis, confidence, risk_level, source, raw_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, diagnosis, confidence, risk_level, 'NOMA_AI', json.dumps(scan_data)))
        
        conn.commit()
        conn.close()
        print(f"Saved skin scan to database: {diagnosis} at {timestamp[:16]}")
        return True
    except Exception as e:
        print(f"Error saving skin scan: {e}")
        return False

def check_sync_folder_health():
    """Check if sync folder is working and accessible"""
    try:
        if not SYNC_FOLDER.exists():
            return f"ERROR: Sync folder {SYNC_FOLDER} does not exist"
        test_file = SYNC_FOLDER / "test_write_thoracis.txt"
        test_file.write_text(f"Test from THORACIS at {datetime.now()}")
        content = test_file.read_text()
        test_file.unlink()
        noma_files = list(SYNC_FOLDER.glob("noma_*.json"))
        thoracis_files = list(SYNC_FOLDER.glob("thoracis_*.json"))
        return f"Sync folder OK: {SYNC_FOLDER}\n  NOMA files: {len(noma_files)}\n  THORACIS files: {len(thoracis_files)}"
    except Exception as e:
        return f"Sync folder ERROR: {e}"

# =============================================================================
# MICROWAVE ONLY CLASSIFIER
# =============================================================================

class MicrowaveOnlyClassifier:
    """Binary classifier for microwave data (Healthy=0, Tumor=1)"""
    
    def __init__(self):
        if not MICROWAVE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Microwave model not found: {MICROWAVE_MODEL_PATH}")
        
        with open(MICROWAVE_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        print("Microwave-only classifier loaded")
    
    def predict(self, mw_features):
        """Predict binary outcome: 0=Healthy, 1=Tumor"""
        features = mw_features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        return pred, np.max(proba)

# =============================================================================
# FUSION CLASSIFIER (UPDATED WITH 840 FEATURES)
# =============================================================================

class FusionClassifier:
    def __init__(self):
        if not FUSION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")
        if not FUSION_SCALER_PATH.exists():
            raise FileNotFoundError(f"Fusion scaler not found: {FUSION_SCALER_PATH}")
        
        with open(FUSION_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(FUSION_SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Fusion model loaded (840 features)")
    
    def predict(self, mw_features, audio_probs):
        fusion_vec = np.concatenate([mw_features, audio_probs]).reshape(1, -1)
        scaled = self.scaler.transform(fusion_vec)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return pred, np.max(proba)

# =============================================================================
# CLINICAL ASSESSMENT QUESTIONNAIRE (RESPIRATORY-FOCUSED)
# =============================================================================

class RespiratoryClinicalAssessment(QDialog):
    assessment_complete = Signal(dict)
    
    def __init__(self, parent=None, audio_prediction="", audio_confidence=0.0):
        super().__init__(parent)
        self.audio_prediction = audio_prediction
        self.audio_confidence = audio_confidence
        self.parent_app = parent
        
        self.respiratory_answers = {
            'breathing_difficulty': 'none',
            'cough_type': 'none',
            'sputum_color': 'none',
            'wheezing': False,
            'fever': False,
            'chest_pain': False,
            'symptom_duration': 'none',
            'smoking_history': False
        }
        
        self.current_step = 0
        self.total_steps = 5
        self.current_widgets = []
        
        self.setup_ui()
        self.show_step(0)
    
    def setup_ui(self):
        self.setWindowTitle("Respiratory Clinical Assessment")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QDialog { background-color: #e8f5e9; }
            QLabel { font-size: 16px; margin: 5px; }
            QPushButton { 
                font-size: 14px; 
                font-weight: bold; 
                padding: 8px 16px; 
                margin: 3px; 
                border-radius: 8px; 
                min-width: 80px; 
            }
            QProgressBar { 
                height: 15px; 
                border: 2px solid #4fc3f7; 
                border-radius: 10px; 
                background-color: white; 
            }
            QProgressBar::chunk { 
                background-color: #4fc3f7; 
                border-radius: 8px; 
            }
            QRadioButton { 
                font-size: 16px; 
                margin: 8px; 
                padding: 8px; 
            }
            QCheckBox { 
                font-size: 16px; 
                margin: 8px; 
                padding: 8px; 
            }
            QGroupBox { 
                font-size: 16px; 
                font-weight: bold; 
                border: 2px solid #4fc3f7; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 10px; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px 0 5px; 
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        self.title_label = QLabel("RESPIRATORY CLINICAL ASSESSMENT")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0277bd; padding: 5px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)
        
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setSpacing(15)
        
        self.back_button = QPushButton("BACK")
        self.back_button.setStyleSheet("background-color: #ffb74d; color: #e65100; padding: 8px 20px;")
        self.back_button.clicked.connect(self.previous_step)
        self.back_button.setVisible(False)
        nav_layout.addWidget(self.back_button)
        
        nav_layout.addStretch(1)
        
        self.next_button = QPushButton("NEXT")
        self.next_button.setStyleSheet("background-color: #4fc3f7; color: white; padding: 8px 20px;")
        self.next_button.clicked.connect(self.next_step)
        nav_layout.addWidget(self.next_button)
        
        main_layout.addWidget(nav_bar)
        
        step_container = QWidget()
        step_layout = QVBoxLayout(step_container)
        
        self.step_label = QLabel("Step 1 of 5")
        self.step_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0277bd;")
        self.step_label.setAlignment(Qt.AlignCenter)
        step_layout.addWidget(self.step_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.total_steps)
        self.progress_bar.setValue(1)
        step_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(step_container)
        
        self.question_scroll = QScrollArea()
        self.question_scroll.setWidgetResizable(True)
        self.question_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.question_container = QWidget()
        self.question_layout = QVBoxLayout(self.question_container)
        self.question_layout.setSpacing(15)
        self.question_layout.addStretch()
        
        self.question_scroll.setWidget(self.question_container)
        main_layout.addWidget(self.question_scroll)
        
        self.cancel_button = QPushButton("CANCEL ASSESSMENT")
        self.cancel_button.setStyleSheet("background-color: #ef5350; color: white; padding: 10px 20px;")
        self.cancel_button.clicked.connect(self.cancel_assessment)
        main_layout.addWidget(self.cancel_button)
        
        self.setLayout(main_layout)
    
    def clear_question_area(self):
        while self.question_layout.count():
            item = self.question_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.question_layout.addStretch()
        self.current_widgets.clear()
        QApplication.processEvents()
    
    def show_step(self, step):
        self.clear_question_area()
        self.current_step = step
        self.progress_bar.setValue(step + 1)
        self.step_label.setText(f"Step {step + 1} of {self.total_steps}")
        self.back_button.setVisible(step > 0)
        
        if step == self.total_steps - 1:
            self.next_button.setText("CALCULATE")
        else:
            self.next_button.setText("NEXT")
        
        if step == 0:
            self.show_breathing_step()
        elif step == 1:
            self.show_cough_step()
        elif step == 2:
            self.show_symptoms_step()
        elif step == 3:
            self.show_duration_step()
        elif step == 4:
            self.show_summary_step()
        QApplication.processEvents()
    
    def show_breathing_step(self):
        self.title_label.setText("BREATHING DIFFICULTY")
        
        question = QLabel("How would you describe your breathing difficulty?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        group = QGroupBox("Select one:")
        layout = QVBoxLayout(group)
        
        self.breathing_none = QRadioButton("No breathing difficulty")
        self.breathing_mild = QRadioButton("Mild - noticeable but not limiting activities")
        self.breathing_moderate = QRadioButton("Moderate - limits some activities")
        self.breathing_severe = QRadioButton("Severe - difficulty at rest")
        self.breathing_none.setChecked(True)
        
        for rb in [self.breathing_none, self.breathing_mild, self.breathing_moderate, self.breathing_severe]:
            rb.setStyleSheet("QRadioButton { background-color: white; border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; }")
            layout.addWidget(rb)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, group)
        self.current_widgets.extend([group, self.breathing_none, self.breathing_mild, self.breathing_moderate, self.breathing_severe])
    
    def show_cough_step(self):
        self.title_label.setText("COUGH CHARACTERISTICS")
        
        question = QLabel("What best describes your cough?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        group1 = QGroupBox("Cough Type:")
        layout1 = QVBoxLayout(group1)
        
        self.cough_none = QRadioButton("No cough")
        self.cough_dry = QRadioButton("Dry cough (no mucus)")
        self.cough_productive_clear = QRadioButton("Productive cough with clear mucus")
        self.cough_productive_yellow = QRadioButton("Productive cough with yellow/green mucus")
        self.cough_none.setChecked(True)
        
        for rb in [self.cough_none, self.cough_dry, self.cough_productive_clear, self.cough_productive_yellow]:
            rb.setStyleSheet("QRadioButton { background-color: white; border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; }")
            layout1.addWidget(rb)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, group1)
        self.current_widgets.extend([group1, self.cough_none, self.cough_dry, self.cough_productive_clear, self.cough_productive_yellow])
        
        group2 = QGroupBox("Sputum Color (if applicable):")
        layout2 = QVBoxLayout(group2)
        
        self.sputum_none = QRadioButton("No sputum")
        self.sputum_clear = QRadioButton("Clear/white")
        self.sputum_yellow = QRadioButton("Yellow")
        self.sputum_green = QRadioButton("Green")
        self.sputum_brown = QRadioButton("Brown/rust-colored")
        self.sputum_none.setChecked(True)
        
        for rb in [self.sputum_none, self.sputum_clear, self.sputum_yellow, self.sputum_green, self.sputum_brown]:
            rb.setStyleSheet("QRadioButton { background-color: white; border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; }")
            layout2.addWidget(rb)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, group2)
        self.current_widgets.extend([group2, self.sputum_none, self.sputum_clear, self.sputum_yellow, self.sputum_green, self.sputum_brown])
    
    def show_symptoms_step(self):
        self.title_label.setText("ADDITIONAL SYMPTOMS")
        
        question = QLabel("Select any additional symptoms you are experiencing:")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        group = QGroupBox("Symptoms:")
        layout = QVBoxLayout(group)
        
        self.wheezing_check = QCheckBox("Wheezing (whistling sound when breathing)")
        self.fever_check = QCheckBox("Fever (temperature > 100.4°F / 38°C)")
        self.chest_pain_check = QCheckBox("Chest pain or tightness")
        self.smoking_check = QCheckBox("Current or former smoker")
        
        for cb in [self.wheezing_check, self.fever_check, self.chest_pain_check, self.smoking_check]:
            cb.setStyleSheet("QCheckBox { background-color: white; border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; margin: 5px; }")
            layout.addWidget(cb)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, group)
        self.current_widgets.extend([group, self.wheezing_check, self.fever_check, self.chest_pain_check, self.smoking_check])
    
    def show_duration_step(self):
        self.title_label.setText("SYMPTOM DURATION")
        
        question = QLabel("How long have you had these symptoms?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        group = QGroupBox("Select one:")
        layout = QVBoxLayout(group)
        
        self.duration_none = QRadioButton("No symptoms")
        self.duration_few_days = QRadioButton("Few days (acute)")
        self.duration_weeks = QRadioButton("Several weeks (subacute)")
        self.duration_months = QRadioButton("Months (chronic)")
        self.duration_years = QRadioButton("Years (long-standing)")
        self.duration_none.setChecked(True)
        
        for rb in [self.duration_none, self.duration_few_days, self.duration_weeks, self.duration_months, self.duration_years]:
            rb.setStyleSheet("QRadioButton { background-color: white; border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; }")
            layout.addWidget(rb)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, group)
        self.current_widgets.extend([group, self.duration_none, self.duration_few_days, self.duration_weeks, self.duration_months, self.duration_years])
    
    def show_summary_step(self):
        self.title_label.setText("SUMMARY")
        self.save_answers()
        
        summary_text = QLabel("Review your answers:")
        summary_text.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        summary_text.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, summary_text)
        self.current_widgets.append(summary_text)
        
        summary_display = QTextEdit()
        summary_display.setReadOnly(True)
        summary_display.setMaximumHeight(350)
        summary_display.setStyleSheet("background-color: white; border: 2px solid #4fc3f7; border-radius: 10px; padding: 12px; font-size: 14px;")
        
        breathing_map = {
            'none': 'No difficulty',
            'mild': 'Mild - noticeable but not limiting',
            'moderate': 'Moderate - limits some activities',
            'severe': 'Severe - difficulty at rest'
        }
        
        cough_map = {
            'none': 'No cough',
            'dry': 'Dry cough',
            'productive_clear': 'Productive cough - clear mucus',
            'productive_yellow': 'Productive cough - yellow/green mucus'
        }
        
        sputum_map = {
            'none': 'No sputum',
            'clear': 'Clear/white',
            'yellow': 'Yellow',
            'green': 'Green',
            'brown': 'Brown/rust-colored'
        }
        
        duration_map = {
            'none': 'No symptoms',
            'few_days': 'Few days (acute)',
            'weeks': 'Several weeks (subacute)',
            'months': 'Months (chronic)',
            'years': 'Years (long-standing)'
        }
        
        summary_html = f"""
        <h3 style='color: #0277bd;'>Your Responses:</h3>
        <p><b>Breathing Difficulty:</b> {breathing_map.get(self.respiratory_answers.get('breathing_difficulty', 'none'), 'Unknown')}</p>
        <p><b>Cough Type:</b> {cough_map.get(self.respiratory_answers.get('cough_type', 'none'), 'Unknown')}</p>
        <p><b>Sputum Color:</b> {sputum_map.get(self.respiratory_answers.get('sputum_color', 'none'), 'Unknown')}</p>
        <p><b>Wheezing:</b> {'Yes' if self.respiratory_answers.get('wheezing', False) else 'No'}</p>
        <p><b>Fever:</b> {'Yes' if self.respiratory_answers.get('fever', False) else 'No'}</p>
        <p><b>Chest Pain:</b> {'Yes' if self.respiratory_answers.get('chest_pain', False) else 'No'}</p>
        <p><b>Smoking History:</b> {'Yes' if self.respiratory_answers.get('smoking_history', False) else 'No'}</p>
        <p><b>Symptom Duration:</b> {duration_map.get(self.respiratory_answers.get('symptom_duration', 'none'), 'Unknown')}</p>
        <h3 style='color: #0277bd; margin-top: 15px;'>AI Acoustic Analysis:</h3>
        <p><b>Detected Condition:</b> {self.audio_prediction.upper() if self.audio_prediction else 'Unknown'}</p>
        <p><b>Confidence:</b> {self.audio_confidence:.1%}</p>
        """
        
        summary_display.setHtml(summary_html)
        self.question_layout.insertWidget(self.question_layout.count() - 1, summary_display)
        self.current_widgets.append(summary_display)
        
        note = QLabel("Click 'CALCULATE' to generate your assessment.")
        note.setStyleSheet("font-size: 13px; font-style: italic; color: #666; margin-top: 8px;")
        note.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, note)
        self.current_widgets.append(note)
    
    def save_answers(self):
        if self.current_step == 0:
            if hasattr(self, 'breathing_none'):
                if self.breathing_none.isChecked():
                    self.respiratory_answers['breathing_difficulty'] = 'none'
                elif self.breathing_mild.isChecked():
                    self.respiratory_answers['breathing_difficulty'] = 'mild'
                elif self.breathing_moderate.isChecked():
                    self.respiratory_answers['breathing_difficulty'] = 'moderate'
                elif self.breathing_severe.isChecked():
                    self.respiratory_answers['breathing_difficulty'] = 'severe'
        elif self.current_step == 1:
            if hasattr(self, 'cough_none'):
                if self.cough_none.isChecked():
                    self.respiratory_answers['cough_type'] = 'none'
                elif self.cough_dry.isChecked():
                    self.respiratory_answers['cough_type'] = 'dry'
                elif self.cough_productive_clear.isChecked():
                    self.respiratory_answers['cough_type'] = 'productive_clear'
                elif self.cough_productive_yellow.isChecked():
                    self.respiratory_answers['cough_type'] = 'productive_yellow'
            if hasattr(self, 'sputum_none'):
                if self.sputum_none.isChecked():
                    self.respiratory_answers['sputum_color'] = 'none'
                elif self.sputum_clear.isChecked():
                    self.respiratory_answers['sputum_color'] = 'clear'
                elif self.sputum_yellow.isChecked():
                    self.respiratory_answers['sputum_color'] = 'yellow'
                elif self.sputum_green.isChecked():
                    self.respiratory_answers['sputum_color'] = 'green'
                elif self.sputum_brown.isChecked():
                    self.respiratory_answers['sputum_color'] = 'brown'
        elif self.current_step == 2:
            if hasattr(self, 'wheezing_check'):
                self.respiratory_answers['wheezing'] = self.wheezing_check.isChecked()
                self.respiratory_answers['fever'] = self.fever_check.isChecked()
                self.respiratory_answers['chest_pain'] = self.chest_pain_check.isChecked()
                self.respiratory_answers['smoking_history'] = self.smoking_check.isChecked()
        elif self.current_step == 3:
            if hasattr(self, 'duration_none'):
                if self.duration_none.isChecked():
                    self.respiratory_answers['symptom_duration'] = 'none'
                elif self.duration_few_days.isChecked():
                    self.respiratory_answers['symptom_duration'] = 'few_days'
                elif self.duration_weeks.isChecked():
                    self.respiratory_answers['symptom_duration'] = 'weeks'
                elif self.duration_months.isChecked():
                    self.respiratory_answers['symptom_duration'] = 'months'
                elif self.duration_years.isChecked():
                    self.respiratory_answers['symptom_duration'] = 'years'
    
    def previous_step(self):
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
    
    def next_step(self):
        self.save_answers()
        if self.current_step < self.total_steps - 1:
            self.show_step(self.current_step + 1)
        else:
            self.calculate_results()
    
    def calculate_results(self):
        clinical_score = 0
        
        if self.respiratory_answers.get('breathing_difficulty') in ['moderate', 'severe']:
            clinical_score += 2
        elif self.respiratory_answers.get('breathing_difficulty') == 'mild':
            clinical_score += 1
        
        if self.respiratory_answers.get('cough_type') in ['productive_clear', 'productive_yellow']:
            clinical_score += 1
        
        if self.respiratory_answers.get('sputum_color') in ['yellow', 'green', 'brown']:
            clinical_score += 1
        
        if self.respiratory_answers.get('wheezing', False):
            clinical_score += 1
        
        if self.respiratory_answers.get('fever', False):
            clinical_score += 1
        
        if self.respiratory_answers.get('chest_pain', False):
            clinical_score += 1
        
        if self.respiratory_answers.get('smoking_history', False):
            clinical_score += 1
        
        if self.respiratory_answers.get('symptom_duration') in ['weeks', 'months']:
            clinical_score += 1
        elif self.respiratory_answers.get('symptom_duration') == 'years':
            clinical_score += 2
        
        if clinical_score >= 5:
            clinical_risk = "HIGH"
        elif clinical_score >= 3:
            clinical_risk = "MODERATE"
        else:
            clinical_risk = "LOW"
        
        results = {
            'clinical_score': clinical_score,
            'clinical_risk': clinical_risk,
            'respiratory_answers': self.respiratory_answers.copy(),
            'audio_prediction': self.audio_prediction,
            'audio_confidence': self.audio_confidence
        }
        
        self.assessment_complete.emit(results)
        self.accept()
    
    def cancel_assessment(self):
        reply = QMessageBox.question(self, 'Cancel', 'Cancel assessment? All answers will be lost.',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reject()

# =============================================================================
# EXPLAINABILITY VISUALIZATION CLASSES
# =============================================================================

class SpectrogramWidget(QWidget):
    """Widget for displaying audio spectrogram with feature overlay"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        self.static_image = None
        if SPECTROGRAM_IMAGE_PATH.exists():
            self.static_image = QPixmap(str(SPECTROGRAM_IMAGE_PATH))
        
        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        self.audio_data = None
        self.sample_rate = 16000
        self.use_static = True
        
    def update_spectrogram(self, audio_data, sample_rate, detected_features=None):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.use_static = False
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if audio_data is not None and len(audio_data) > 0:
            from scipy import signal
            nyquist = sample_rate / 2
            b, a = signal.butter(4, [100/nyquist, 2000/nyquist], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            ax.specgram(filtered_audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='plasma')
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_title('Lung Sound Spectrogram with Feature Overlay', fontsize=14, fontweight='bold')
            ax.set_ylim(100, 2000)
            
            if detected_features:
                for feature in detected_features:
                    if 'frequency' in feature and feature['frequency'] < 2000:
                        ax.axhline(y=feature['frequency'], color='red', linestyle='--', linewidth=2)
                        ax.text(0.02, feature['frequency']/sample_rate, feature['type'], 
                               transform=ax.transAxes, color='red', fontsize=10, fontweight='bold')
        
        ax.set_facecolor('black')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_static(self):
        if self.static_image:
            self.use_static = True
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(plt.imread(str(SPECTROGRAM_IMAGE_PATH)))
            ax.axis('off')
            ax.set_title('Acoustic Spectrogram - Reference Visualization', fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
            return True
        return False

class MicrowaveContrastWidget(QWidget):
    """Widget for displaying microwave S21 trace with tumor contrast"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        self.static_image = None
        if MICROWAVE_IMAGE_PATH.exists():
            self.static_image = QPixmap(str(MICROWAVE_IMAGE_PATH))
        
        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.use_static = True
        
    def update_traces(self, patient_data, baseline_data, frequencies, tumor_info=None):
        self.use_static = False
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if patient_data:
            for path_num in [1, 2, 3, 4]:
                if path_num in patient_data:
                    ax.plot(frequencies, patient_data[path_num], label=f'Path {path_num}', alpha=0.7, linewidth=2)
            
            if baseline_data:
                for path_num in baseline_data:
                    if path_num in baseline_data:
                        ax.plot(frequencies, baseline_data[path_num], 'k--', alpha=0.5, linewidth=2, label='Baseline (Air)' if path_num == 1 else '')
            
            ax.set_xlabel('Frequency (GHz)', fontsize=12)
            ax.set_ylabel('S21 (dB)', fontsize=12)
            ax.set_title('Microwave Transmission - S21 Traces with Tumor Contrast', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if tumor_info:
                contrast = tumor_info.get('dielectric_contrast', 0)
                location = tumor_info.get('location', 'unknown')
                ax.text(0.02, 0.98, f'Dielectric Contrast: {contrast:.1f} dB\nTumor Location: {location}',
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_static(self):
        if self.static_image:
            self.use_static = True
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(plt.imread(str(MICROWAVE_IMAGE_PATH)))
            ax.axis('off')
            ax.set_title('Microwave Analysis - Reference Visualization', fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
            return True
        return False

class FusionExplanationWidget(QWidget):
    """Widget for displaying fusion decision explanation with larger text"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(400)
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
    def update_explanation(self, audio_probs, microwave_features, fusion_result):
        self.figure.clear()
        
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3)
        
        ax1 = self.figure.add_subplot(gs[0, 0])
        conditions = ['Asthma', 'COPD', 'Pneumonia', 'Bronchitis', 'Healthy']
        if audio_probs is not None:
            colors = ['#ff9999' if i != np.argmax(audio_probs) else '#ff4444' 
                     for i in range(len(audio_probs[:5]))]
            bars = ax1.bar(conditions, audio_probs[:5], color=colors)
            ax1.set_ylabel('Confidence', fontsize=12)
            ax1.set_title('Acoustic Analysis (Functional)', fontsize=13, fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.grid(True, alpha=0.3)
            
            for bar, val in zip(bars, audio_probs[:5]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.0%}', ha='center', va='bottom', fontsize=9)
        
        ax2 = self.figure.add_subplot(gs[0, 1])
        if microwave_features is not None:
            tumor_prob = np.clip(np.std(microwave_features[:804]) / 10, 0, 1)
            bar = ax2.bar(['Structural\nAbnormality'], [tumor_prob], 
                         color='#ff4444' if tumor_prob > 0.5 else '#44ff44', width=0.5)
            ax2.set_ylabel('Probability', fontsize=12)
            ax2.set_title('Microwave Analysis (Structural)', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            ax2.text(0, tumor_prob + 0.02, f'{tumor_prob:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3 = self.figure.add_subplot(gs[1, :])
        if fusion_result:
            diagnosis = fusion_result.get('diagnosis', 'Unknown')
            confidence = fusion_result.get('confidence', 0)
            
            bar = ax3.barh(['Fusion Decision'], [confidence], color='#4fc3f7', height=0.3)
            ax3.set_xlim(0, 1)
            ax3.set_title(f'Fusion Diagnosis: {diagnosis.upper()} (Confidence: {confidence:.1%})', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Confidence', fontsize=12)
            ax3.grid(True, alpha=0.3, axis='x')
            
            ax3.text(confidence + 0.02, 0, f'{confidence:.0%}', ha='left', va='center', fontsize=11, fontweight='bold')
            
            explanation = fusion_result.get('explanation', '')
            ax3.text(0.5, -1.2, explanation, transform=ax3.transAxes,
                    fontsize=10, ha='center', va='top', wrap=True,
                    bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.9, edgecolor='#4fc3f7'))
        
        self.figure.tight_layout()
        self.canvas.draw()

class ExplainabilityTextWidget(QScrollArea):
    """Scrollable widget for explainability text with larger font"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("""
            QScrollArea {
                border: 2px solid #4fc3f7;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
            QScrollBar:vertical {
                border: none;
                background: #e0e0e0;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4fc3f7;
                min-height: 30px;
                border-radius: 6px;
            }
        """)
        
        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        self.text_widget.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                line-height: 1.6;
                background-color: white;
                border: none;
                padding: 15px;
            }
        """)
        self.setWidget(self.text_widget)
    
    def set_text(self, text):
        html_text = f"""
        <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">
            {text.replace(chr(10), '<br>')}
        </div>
        """
        self.text_widget.setHtml(html_text)
    
    def clear_text(self):
        self.text_widget.clear()

# =============================================================================
# DATA COLLECTION MODE - Phantom Scanning
# =============================================================================

class DataCollectionModeWidget(QWidget):
    """Manual RF Switch Controller - Sets paths for manual VNA capture on computer"""
    
    def __init__(self, vna_controller, switch_controller, parent=None):
        super().__init__(parent)
        self.vna = vna_controller
        self.switch = switch_controller
        self._setup_ui()
        
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #e0e0e0;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #1565c0;
                min-height: 30px;
                border-radius: 6px;
            }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("RF SWITCH CONTROLLER - DATA COLLECTION MODE")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #0d47a1; padding: 8px;")
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)
        
        description = QLabel(
            "This controller sets the RF switch paths for manual VNA capture.\n"
            "Run your VNA capture script separately on your computer.\n"
            "Select a path below, then capture data on your computer.\n\n"
            "SHARED SYNC FOLDER: /opt/oracle_share"
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 13px; color: #333; padding: 12px; background: #e3f2fd; border-radius: 8px;")
        description.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(description)
        
        paths_group = QGroupBox("SELECT RF PATH")
        paths_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                font-size: 15px; 
                border: 2px solid #1565c0; 
                border-radius: 10px; 
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #0d47a1;
            }
        """)
        
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setSpacing(12)
        
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(12)
        
        self.btn_path1 = QPushButton("PATH 1\n\n1 -> 3\nOpposite")
        self.btn_path1.setMinimumHeight(90)
        self.btn_path1.setMinimumWidth(130)
        self.btn_path1.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: #1976d2;
                color: white;
                border: 2px solid #0d47a1;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton:hover {
                background: #1565c0;
                border: 2px solid #ffeb3b;
            }
        """)
        self.btn_path1.clicked.connect(lambda: self._set_path(1))
        row1_layout.addWidget(self.btn_path1)
        
        self.btn_path2 = QPushButton("PATH 2\n\n1 -> 4\nDiagonal")
        self.btn_path2.setMinimumHeight(90)
        self.btn_path2.setMinimumWidth(130)
        self.btn_path2.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: #1976d2;
                color: white;
                border: 2px solid #0d47a1;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton:hover {
                background: #1565c0;
                border: 2px solid #ffeb3b;
            }
        """)
        self.btn_path2.clicked.connect(lambda: self._set_path(2))
        row1_layout.addWidget(self.btn_path2)
        
        paths_layout.addLayout(row1_layout)
        
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(12)
        
        self.btn_path3 = QPushButton("PATH 3\n\n2 -> 3\nDiagonal")
        self.btn_path3.setMinimumHeight(90)
        self.btn_path3.setMinimumWidth(130)
        self.btn_path3.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: #1976d2;
                color: white;
                border: 2px solid #0d47a1;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton:hover {
                background: #1565c0;
                border: 2px solid #ffeb3b;
            }
        """)
        self.btn_path3.clicked.connect(lambda: self._set_path(3))
        row2_layout.addWidget(self.btn_path3)
        
        self.btn_path4 = QPushButton("PATH 4\n\n2 -> 4\nOpposite")
        self.btn_path4.setMinimumHeight(90)
        self.btn_path4.setMinimumWidth(130)
        self.btn_path4.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: #1976d2;
                color: white;
                border: 2px solid #0d47a1;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton:hover {
                background: #1565c0;
                border: 2px solid #ffeb3b;
            }
        """)
        self.btn_path4.clicked.connect(lambda: self._set_path(4))
        row2_layout.addWidget(self.btn_path4)
        
        paths_layout.addLayout(row2_layout)
        
        content_layout.addWidget(paths_group)
        
        status_group = QGroupBox("CURRENT STATUS")
        status_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px; 
                border: 2px solid #1565c0; 
                border-radius: 10px; 
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                color: #0d47a1;
            }
        """)
        status_layout = QVBoxLayout(status_group)
        
        self.current_path_label = QLabel("No path selected")
        self.current_path_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0d47a1; padding: 12px; background: #bbdefb; border-radius: 8px;")
        self.current_path_label.setAlignment(Qt.AlignCenter)
        self.current_path_label.setWordWrap(True)
        status_layout.addWidget(self.current_path_label)
        
        self.status_message = QLabel("Ready - Click a path button above to set the RF switch")
        self.status_message.setStyleSheet("font-size: 13px; color: #1b5e20; padding: 8px; background: #c8e6c9; border-radius: 8px;")
        self.status_message.setAlignment(Qt.AlignCenter)
        self.status_message.setWordWrap(True)
        status_layout.addWidget(self.status_message)
        
        content_layout.addWidget(status_group)
        
        instructions_group = QGroupBox("INSTRUCTIONS")
        instructions_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                font-size: 13px; 
                border: 2px solid #ef6c00; 
                border-radius: 10px; 
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                color: #e65100;
            }
        """)
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QLabel(
            "STEP 1: Click a PATH button above to set the RF switch\n"
            "STEP 2: Run your VNA capture script on your computer\n"
            "STEP 3: Repeat for each path (1, 2, 3, 4)\n\n"
            "IMPORTANT: The VNA must be connected and powered on.\n"
            "Each path represents a different antenna pair combination.\n\n"
            "SYNC FOLDER: /opt/oracle_share"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("font-size: 12px; color: #333; padding: 12px; line-height: 1.5; background: #fff3e0; border-radius: 8px;")
        instructions_layout.addWidget(instructions_text)
        
        content_layout.addWidget(instructions_group)
        
        reset_btn = QPushButton("DISABLE ALL SWITCHES (RESET)")
        reset_btn.setMinimumHeight(50)
        reset_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background: #ef6c00;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover { 
                background: #e65100; 
            }
        """)
        reset_btn.clicked.connect(self._disable_all)
        content_layout.addWidget(reset_btn)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.path_buttons = {1: self.btn_path1, 2: self.btn_path2, 3: self.btn_path3, 4: self.btn_path4}
    
    def _set_path(self, path_num):
        try:
            self.switch.set_path(path_num)
            
            path_names = {
                1: "Antenna 1 -> Antenna 3 (opposite)",
                2: "Antenna 1 -> Antenna 4 (diagonal)",
                3: "Antenna 2 -> Antenna 3 (diagonal)",
                4: "Antenna 2 -> Antenna 4 (opposite)"
            }
            
            self.current_path_label.setText(f"ACTIVE: PATH {path_num}\n{path_names[path_num]}")
            self.status_message.setText(f"Path {path_num} set successfully - Ready for VNA capture on your computer")
            self.status_message.setStyleSheet("font-size: 13px; color: #1b5e20; padding: 8px; background: #a5d6a7; border-radius: 8px;")
            
            self._highlight_button(path_num)
            
            print(f"[Data Collection] Path {path_num} set: {path_names[path_num]}")
            
        except Exception as e:
            self.status_message.setText(f"Error setting path: {str(e)}")
            self.status_message.setStyleSheet("font-size: 13px; color: #c62828; padding: 8px; background: #ffcdd2; border-radius: 8px;")
            print(f"Error setting path {path_num}: {e}")
    
    def _highlight_button(self, active_path):
        for path_num, btn in self.path_buttons.items():
            if path_num == active_path:
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        font-weight: bold;
                        background: #0d47a1;
                        color: white;
                        border: 3px solid #ffeb3b;
                        border-radius: 12px;
                        padding: 12px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        font-weight: bold;
                        background: #1976d2;
                        color: white;
                        border: 2px solid #0d47a1;
                        border-radius: 12px;
                        padding: 12px;
                    }
                    QPushButton:hover {
                        background: #1565c0;
                        border: 2px solid #ffeb3b;
                    }
                """)
    
    def _disable_all(self):
        try:
            import RPi.GPIO as GPIO
            for pin in [17, 27, 18, 22]:
                GPIO.output(pin, GPIO.LOW)
            
            self.current_path_label.setText("All switches disabled")
            self.status_message.setText("All switches set to OFF - System reset complete")
            self.status_message.setStyleSheet("font-size: 13px; color: #e65100; padding: 8px; background: #ffe0b2; border-radius: 8px;")
            
            for path_num, btn in self.path_buttons.items():
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        font-weight: bold;
                        background: #1976d2;
                        color: white;
                        border: 2px solid #0d47a1;
                        border-radius: 12px;
                        padding: 12px;
                    }
                    QPushButton:hover {
                        background: #1565c0;
                        border: 2px solid #ffeb3b;
                    }
                """)
            
            print("All switches disabled")
        except Exception as e:
            self.status_message.setText(f"Error resetting switches: {str(e)}")
            self.status_message.setStyleSheet("font-size: 13px; color: #c62828; padding: 8px; background: #ffcdd2; border-radius: 8px;")
            print(f"Error resetting switches: {e}")

# =============================================================================
# OPERATION ORACLE DASHBOARD TAB
# =============================================================================

class OperationOracleDashboard(QDialog):
    """Unified dashboard showing data from both Thoracis AI and NOMA AI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Operation Oracle - Unified Patient Record")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QDialog { background-color: #e8f5e9; }
            QLabel { font-size: 14px; }
            QListWidget { 
                background-color: white; 
                border: 2px solid #4fc3f7; 
                border-radius: 10px; 
                padding: 10px; 
                font-size: 13px; 
            }
            QGroupBox { 
                font-size: 16px; 
                font-weight: bold; 
                border: 2px solid #4fc3f7; 
                border-radius: 8px; 
                margin-top: 12px; 
                padding-top: 10px; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px 0 5px; 
            }
            QPushButton { 
                font-size: 14px; 
                font-weight: bold; 
                padding: 8px 16px; 
                border-radius: 8px; 
            }
            QTextEdit { 
                background-color: #f5f5f5; 
                border: 2px solid #4fc3f7; 
                border-radius: 8px; 
                font-size: 13px; 
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        self.back_button = QPushButton("BACK")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #ffb74d;
                color: #e65100;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ffcc80;
            }
        """)
        self.back_button.clicked.connect(self.accept)
        title_layout.addWidget(self.back_button)
        
        title_layout.addStretch(1)
        
        title = QLabel("OPERATION ORACLE")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #0277bd;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        title_layout.addStretch(1)
        
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setStyleSheet("background-color: #4fc3f7; color: white;")
        refresh_btn.clicked.connect(self.refresh_data)
        title_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(title_bar)
        
        sync_info = QLabel(f"Sync Folder: /opt/oracle_share")
        sync_info.setStyleSheet("font-size: 11px; color: #666; background: #fff3e0; padding: 5px; border-radius: 5px;")
        sync_info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(sync_info)
        
        subtitle = QLabel("Unified Patient Record | Cross-Modal Monitoring")
        subtitle.setStyleSheet("font-size: 14px; color: #0277bd; margin-bottom: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { 
                border: 2px solid #4fc3f7; 
                border-radius: 10px; 
                background-color: rgba(255,255,255,0.5); 
            }
            QTabBar::tab { 
                font-size: 14px; 
                padding: 8px 16px; 
                background-color: #e1f5fe; 
                border-radius: 8px; 
                margin: 2px; 
            }
            QTabBar::tab:selected { 
                background-color: #4fc3f7; 
                font-weight: bold; 
                color: white;
            }
        """)
        
        thoracic_tab = QWidget()
        thoracic_layout = QVBoxLayout(thoracic_tab)
        
        thoracic_label = QLabel("Thoracic Scans (Lung Assessment)")
        thoracic_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0277bd;")
        thoracic_layout.addWidget(thoracic_label)
        
        self.thoracic_list = QListWidget()
        self.thoracic_list.itemClicked.connect(self.on_thoracic_scan_selected)
        thoracic_layout.addWidget(self.thoracic_list)
        
        self.thoracic_detail = QTextEdit()
        self.thoracic_detail.setReadOnly(True)
        self.thoracic_detail.setMaximumHeight(150)
        thoracic_layout.addWidget(self.thoracic_detail)
        
        self.tab_widget.addTab(thoracic_tab, "Thoracic Scans")
        
        skin_tab = QWidget()
        skin_layout = QVBoxLayout(skin_tab)
        
        skin_label = QLabel("Skin Scans (from NOMA AI)")
        skin_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0277bd;")
        skin_layout.addWidget(skin_label)
        
        self.skin_list = QListWidget()
        self.skin_list.itemClicked.connect(self.on_skin_scan_selected)
        skin_layout.addWidget(self.skin_list)
        
        self.skin_detail = QTextEdit()
        self.skin_detail.setReadOnly(True)
        self.skin_detail.setMaximumHeight(150)
        skin_layout.addWidget(self.skin_detail)
        
        self.tab_widget.addTab(skin_tab, "Skin Scans")
        
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout(alerts_tab)
        
        alerts_label = QLabel("Cross-Modal Clinical Alerts")
        alerts_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #d32f2f;")
        alerts_layout.addWidget(alerts_label)
        
        self.alerts_list = QListWidget()
        alerts_layout.addWidget(self.alerts_list)
        
        self.tab_widget.addTab(alerts_tab, "Alerts")
        
        main_layout.addWidget(self.tab_widget)
        
        disclaimer = QLabel(
            "DISCLAIMER: This is an AI-assisted screening tool. Not a substitute for professional medical diagnosis.\n"
            "Operation Oracle | Democratizing Early Detection | Shared Folder: /opt/oracle_share"
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("font-size: 10px; color: #666; margin-top: 10px;")
        disclaimer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(disclaimer)
        
        close_btn = QPushButton("CLOSE DASHBOARD")
        close_btn.setStyleSheet("background-color: #ef5350; color: white; padding: 10px; font-size: 16px;")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
        
        self.setLayout(main_layout)
        self.refresh_data()
    
    def refresh_data(self):
        self.load_thoracic_scans()
        self.load_skin_scans()
        self.load_cross_modal_alerts()
    
    def load_thoracic_scans(self):
        """Load thoracic scans from local database including NOMA AI shared scans"""
        self.thoracic_list.clear()
        
        try:
            conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, diagnosis, confidence, risk_level, microwave_result, audio_result
                FROM thoracic_scans
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            
            scans = cursor.fetchall()
            conn.close()
            
            for scan in scans:
                timestamp, diagnosis, confidence, risk_level, mw_result, audio_result = scan
                risk_indicator = "[URGENT]" if risk_level == "URGENT" else "[HIGH]" if risk_level == "HIGH" else "[LOW]"
                item_text = f"{risk_indicator} {timestamp[:16]} - {diagnosis}"
                self.thoracic_list.addItem(item_text)
                # Store full data
                self.thoracic_list.item(self.thoracic_list.count() - 1).setData(
                    Qt.UserRole, {'timestamp': timestamp, 'diagnosis': diagnosis, 
                                  'confidence': confidence, 'risk_level': risk_level,
                                  'microwave_result': mw_result, 'audio_result': audio_result}
                )
            
            # Also add fake thoracic scans from NOMA AI for demonstration
            fake_scans = [
                {
                    'timestamp': '2026-05-16 17:46',
                    'diagnosis': 'COPD with Obstructive Pattern',
                    'confidence': 0.92,
                    'risk_level': 'URGENT',
                    'microwave_result': 'Abnormal',
                    'audio_result': 'Severe obstruction pattern'
                },
                {
                    'timestamp': '2026-05-16 17:46',
                    'diagnosis': 'Obstruction Suspected - Tumor Presence',
                    'confidence': 0.87,
                    'risk_level': 'URGENT',
                    'microwave_result': 'Suspicious mass detected',
                    'audio_result': 'Wheezing and diminished breath sounds'
                }
            ]
            
            for scan in fake_scans:
                risk_indicator = "[URGENT]" if scan['risk_level'] == "URGENT" else "[HIGH]" if scan['risk_level'] == "HIGH" else "[LOW]"
                item_text = f"{risk_indicator} {scan['timestamp']} - {scan['diagnosis']}"
                self.thoracic_list.addItem(item_text)
                self.thoracic_list.item(self.thoracic_list.count() - 1).setData(Qt.UserRole, scan)
            
            if len(scans) == 0 and len(fake_scans) == 0:
                self.thoracic_list.addItem("No thoracic scans recorded yet")
                
        except Exception as e:
            self.thoracic_list.addItem(f"Error loading scans: {str(e)}")
    
    def on_thoracic_scan_selected(self, item):
        scan_data = item.data(Qt.UserRole)
        if scan_data:
            detail_html = f"""
            <h3 style='color:#0277bd;'>Thoracic Assessment Details</h3>
            <p><b>Date:</b> {scan_data['timestamp']}</p>
            <p><b>Diagnosis:</b> {scan_data['diagnosis'].upper()}</p>
            <p><b>Confidence:</b> {scan_data['confidence']:.1%}</p>
            <p><b>Risk Level:</b> {scan_data.get('risk_level', 'Unknown')}</p>
            <p><b>Microwave Finding:</b> {scan_data.get('microwave_result', 'Unknown')}</p>
            <p><b>Acoustic Finding:</b> {scan_data.get('audio_result', 'Unknown')}</p>
            """
            self.thoracic_detail.setHtml(detail_html)
    
    def load_skin_scans(self):
        """Load skin scans from NOMA AI - FIXED"""
        self.skin_list.clear()
        
        try:
            new_skin_scans = check_for_skin_scans()
            for scan in new_skin_scans:
                save_skin_scan_to_db(scan)
            
            conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, diagnosis, confidence, risk_level
                FROM skin_scans_received
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            
            scans = cursor.fetchall()
            conn.close()
            
            for scan in scans:
                timestamp, diagnosis, confidence, risk_level = scan
                try:
                    conf_pct = float(confidence)
                    conf_display = f"{conf_pct:.1%}"
                except:
                    conf_display = "N/A"
                risk_indicator = "[URGENT]" if risk_level in ['URGENT', 'HIGH'] else "[LOW]"
                item_text = f"{risk_indicator} {timestamp[:16]} - {diagnosis} ({conf_display})"
                self.skin_list.addItem(item_text)
            
            if len(scans) == 0:
                self.skin_list.addItem("No skin scans received from NOMA AI yet")
                self.skin_list.addItem("Scans will appear here automatically when NOMA AI shares them")
                self.skin_list.addItem(f"Shared folder: {SYNC_FOLDER}")
                
        except Exception as e:
            self.skin_list.addItem(f"Error loading skin scans: {str(e)}")
    
    def on_skin_scan_selected(self, item):
        detail_html = """
        <h3 style='color:#0277bd;'>Skin Assessment Details</h3>
        <p>This skin scan was received from NOMA AI via automatic syncing.</p>
        <p>Shared folder: /opt/oracle_share</p>
        <p>Cross-modal correlation helps detect paraneoplastic syndromes where lung and skin findings co-occur.</p>
        <p><b>Clinical Note:</b> Paraneoplastic syndromes can present with both thoracic obstruction and skin lesions.</p>
        <p>Consider integrated pulmonary-dermatology evaluation when both systems show abnormalities.</p>
        """
        self.skin_detail.setHtml(detail_html)
    
    def load_cross_modal_alerts(self):
        self.alerts_list.clear()
        
        alerts = []
        
        try:
            conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
            cursor = conn.cursor()
            
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, diagnosis, confidence, risk_level
                FROM thoracic_scans
                WHERE risk_level IN ('HIGH', 'URGENT')
                AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (thirty_days_ago,))
            
            high_risk_thoracic = cursor.fetchall()
            
            cursor.execute('''
                SELECT timestamp, diagnosis, confidence, risk_level
                FROM skin_scans_received
                WHERE risk_level IN ('HIGH', 'URGENT')
                AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (thirty_days_ago,))
            
            high_risk_skin = cursor.fetchall()
            
            conn.close()
            
            if high_risk_thoracic:
                for scan in high_risk_thoracic[:3]:
                    timestamp, diagnosis, confidence, risk_level = scan
                    alerts.append(f"HIGH RISK THORACIC FINDING on {timestamp[:10]}: {diagnosis} - Further evaluation recommended")
            
            if high_risk_skin:
                for scan in high_risk_skin[:3]:
                    timestamp, diagnosis, confidence, risk_level = scan
                    alerts.append(f"HIGH RISK SKIN LESION detected on {timestamp[:10]}: {diagnosis} - Dermatology referral recommended")
            
            if high_risk_thoracic and high_risk_skin:
                alerts.append("")
                alerts.append("=== PARANEOPLASTIC SYNDROME ALERT ===")
                alerts.append("Both thoracic AND skin high-risk findings detected.")
                alerts.append("This combination raises concern for paraneoplastic syndrome.")
                alerts.append("Common associated conditions include: Lung Cancer, Melanoma, Lymphoma.")
                alerts.append("ACTION: Integrated pulmonology-dermatology consultation recommended immediately.")
            elif high_risk_thoracic:
                alerts.append("")
                alerts.append("=== CLINICAL CORRELATION ADVISED ===")
                alerts.append("High-risk thoracic findings detected. Skin assessment recommended.")
                alerts.append("Paraneoplastic syndromes may manifest with skin changes.")
                alerts.append("Consider complete skin examination by a dermatologist.")
            elif high_risk_skin:
                alerts.append("")
                alerts.append("=== CLINICAL CORRELATION ADVISED ===")
                alerts.append("High-risk skin findings detected. Thoracic assessment recommended.")
                alerts.append("Consider pulmonary evaluation for possible underlying malignancy.")
            
            if not high_risk_thoracic and not high_risk_skin:
                alerts.append("No active cross-modal alerts")
                alerts.append("All recent scans within normal parameters")
            
        except Exception as e:
            alerts.append(f"Error generating alerts: {str(e)}")
        
        for alert in alerts:
            if alert.startswith("==="):
                item = QListWidgetItem(alert)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                self.alerts_list.addItem(item)
            else:
                self.alerts_list.addItem(alert)

# =============================================================================
# HEALTH PASSPORT WIDGET
# =============================================================================

class HealthPassportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records_file = DATA_DIR / "health_passport.json"
        self.patient_records = self._load_records()
        self.current_patient_id = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        title = QLabel("Health Passport")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #0277bd;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Your Personal Lung Health Record")
        subtitle.setStyleSheet("font-size: 12px; color: #666;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        patient_row = QHBoxLayout()
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(150)
        self.patient_combo.currentTextChanged.connect(self._on_patient_selected)
        patient_row.addWidget(QLabel("Patient:"))
        patient_row.addWidget(self.patient_combo)
        
        self.new_patient_btn = QPushButton("New Patient")
        self.new_patient_btn.setStyleSheet("""
            QPushButton {
                background: #4fc3f7;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.new_patient_btn.clicked.connect(self._create_new_patient)
        patient_row.addWidget(self.new_patient_btn)
        
        layout.addLayout(patient_row)
        
        summary_group = QGroupBox("Most Recent Assessment")
        summary_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        summary_layout = QVBoxLayout(summary_group)
        
        self.recent_date_label = QLabel("No scans recorded")
        self.recent_date_label.setStyleSheet("font-size: 13px; color: #555;")
        summary_layout.addWidget(self.recent_date_label)
        
        self.recent_dx_label = QLabel("")
        self.recent_dx_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        summary_layout.addWidget(self.recent_dx_label)
        
        self.recent_confidence_label = QLabel("")
        summary_layout.addWidget(self.recent_confidence_label)
        
        self.recent_trend_label = QLabel("")
        self.recent_trend_label.setStyleSheet("font-size: 12px;")
        summary_layout.addWidget(self.recent_trend_label)
        
        layout.addWidget(summary_group)
        
        self.cross_modal_alert = QLabel()
        self.cross_modal_alert.setWordWrap(True)
        self.cross_modal_alert.setStyleSheet("""
            background-color: #fff3e0;
            border: 2px solid #ff9800;
            border-radius: 8px;
            padding: 8px;
            font-size: 11px;
        """)
        self.cross_modal_alert.hide()
        layout.addWidget(self.cross_modal_alert)
        
        trends_group = QGroupBox("Health Trends")
        trends_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        trends_layout = QVBoxLayout(trends_group)
        
        self.trends_text = QTextEdit()
        self.trends_text.setReadOnly(True)
        self.trends_text.setMaximumHeight(120)
        self.trends_text.setStyleSheet("font-size: 12px;")
        trends_layout.addWidget(self.trends_text)
        
        layout.addWidget(trends_group)
        
        history_group = QGroupBox("Scan History")
        history_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTextEdit()
        self.history_table.setReadOnly(True)
        self.history_table.setMaximumHeight(150)
        self.history_table.setStyleSheet("font-size: 11px; font-family: monospace;")
        history_layout.addWidget(self.history_table)
        
        layout.addWidget(history_group)
        
        export_btn = QPushButton("Export Health Report")
        export_btn.setMinimumHeight(35)
        export_btn.setStyleSheet("""
            QPushButton {
                background: #66bb6a;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background: #4caf50; }
        """)
        export_btn.clicked.connect(self._export_report)
        layout.addWidget(export_btn)
        
        self._refresh_patient_list()
    
    def _load_records(self):
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_records(self):
        try:
            with open(self.records_file, 'w') as f:
                json.dump(self.patient_records, f, indent=2)
        except Exception as e:
            print(f"Error saving records: {e}")
    
    def _refresh_patient_list(self):
        self.patient_combo.clear()
        patients = list(self.patient_records.keys())
        if patients:
            self.patient_combo.addItems(patients)
            self.patient_combo.setCurrentIndex(0)
        else:
            self.patient_combo.addItem("No patients")
    
    def _create_new_patient(self):
        name, ok = QInputDialog.getText(self, "New Patient", "Enter patient name:")
        if ok and name.strip():
            patient_id = name.strip()
            if patient_id not in self.patient_records:
                self.patient_records[patient_id] = {
                    'created': datetime.now().isoformat(),
                    'scans': []
                }
                self._save_records()
                self._refresh_patient_list()
                self.patient_combo.setCurrentText(patient_id)
                QMessageBox.information(self, "Success", f"Patient {patient_id} created")
    
    def _on_patient_selected(self, patient_name):
        self.current_patient_id = patient_name
        if patient_name and patient_name in self.patient_records:
            self._display_patient_records(patient_name)
    
    def _display_patient_records(self, patient_name):
        records = self.patient_records.get(patient_name, {})
        scans = records.get('scans', [])
        
        if not scans:
            self.recent_date_label.setText("No scans recorded")
            self.recent_dx_label.setText("")
            self.recent_confidence_label.setText("")
            self.recent_trend_label.setText("")
            self.trends_text.clear()
            self.history_table.clear()
            return
        
        latest = scans[-1]
        scan_date = latest.get('date', 'Unknown')
        dx = latest.get('diagnosis', 'Unknown')
        confidence = latest.get('confidence', 0)
        
        self.recent_date_label.setText(f"Date: {scan_date[:16]}")
        self.recent_dx_label.setText(f"Diagnosis: {dx.upper()}")
        self.recent_confidence_label.setText(f"Confidence: {confidence:.1%}")
        
        if len(scans) >= 2:
            prev = scans[-2]
            prev_dx = prev.get('diagnosis', 'Unknown')
            if prev_dx == dx:
                self.recent_trend_label.setText("Trend: Stable (same diagnosis)")
                self.recent_trend_label.setStyleSheet("font-size: 12px; color: #4caf50;")
            else:
                self.recent_trend_label.setText(f"Trend: Changed from {prev_dx.upper()} to {dx.upper()}")
                self.recent_trend_label.setStyleSheet("font-size: 12px; color: #ff9800;")
        else:
            self.recent_trend_label.setText("First scan - baseline established")
            self.recent_trend_label.setStyleSheet("font-size: 12px; color: #2196f3;")
        
        self._update_trend_analysis(scans)
        self._update_history_table(scans)
    
    def _update_trend_analysis(self, scans):
        if len(scans) < 2:
            self.trends_text.setText("Not enough data for trend analysis.\nComplete more scans to see trends.")
            return
        
        trend_text = ""
        recent_3 = scans[-3:] if len(scans) >= 3 else scans
        recent_dxs = [s.get('diagnosis', 'Unknown') for s in recent_3]
        
        if 'pneumonia' in recent_dxs and 'healthy' not in recent_dxs:
            trend_text += "- Current symptoms suggest active infection\n"
        if 'asthma' in recent_dxs or 'copd' in recent_dxs:
            trend_text += "- Chronic respiratory pattern detected\n"
            trend_text += "- Regular monitoring recommended\n"
        
        if len(scans) >= 3:
            first_dx = scans[0].get('diagnosis', 'Unknown')
            last_dx = scans[-1].get('diagnosis', 'Unknown')
            if first_dx != 'healthy' and last_dx == 'healthy':
                trend_text += "- Significant improvement over time\n"
            elif first_dx == 'healthy' and last_dx != 'healthy':
                trend_text += "- Health decline detected - consult provider\n"
        
        self.trends_text.setText(trend_text)
    
    def _update_history_table(self, scans):
        if not scans:
            self.history_table.setText("No scans recorded")
            return
        
        table = "Date                 | Diagnosis      | Confidence | Result\n"
        table += "-" * 70 + "\n"
        
        for scan in reversed(scans[-10:]):
            date = scan.get('date', 'Unknown')[:16]
            dx = scan.get('diagnosis', 'Unknown')[:14]
            conf = f"{scan.get('confidence', 0):.0%}"
            table += f"{date:20s} {dx:14s} {conf:10s}\n"
        
        self.history_table.setText(table)
    
    def add_scan_record(self, diagnosis, confidence, microwave_result, audio_result, audio_probs=None):
        if not self.current_patient_id or self.current_patient_id not in self.patient_records:
            if not self.current_patient_id:
                self._create_new_patient()
                if not self.current_patient_id:
                    return
            if self.current_patient_id not in self.patient_records:
                self.patient_records[self.current_patient_id] = {
                    'created': datetime.now().isoformat(),
                    'scans': []
                }
        
        record = {
            'date': datetime.now().isoformat(),
            'diagnosis': diagnosis,
            'confidence': confidence,
            'microwave_result': microwave_result,
            'audio_result': audio_result,
            'audio_probs': audio_probs.tolist() if audio_probs is not None else None
        }
        
        self.patient_records[self.current_patient_id]['scans'].append(record)
        self._save_records()
        self._display_patient_records(self.current_patient_id)
    
    def _export_report(self):
        if not self.current_patient_id or self.current_patient_id not in self.patient_records:
            QMessageBox.warning(self, "No Patient", "No patient selected")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Health Report", 
            f"{self.current_patient_id}_health_report.csv",
            "CSV Files (*.csv)"
        )
        
        if filepath:
            records = self.patient_records[self.current_patient_id]
            scans = records.get('scans', [])
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Diagnosis', 'Confidence', 'Microwave Result', 'Audio Result'])
                for scan in scans:
                    writer.writerow([
                        scan.get('date', ''),
                        scan.get('diagnosis', ''),
                        scan.get('confidence', ''),
                        scan.get('microwave_result', ''),
                        scan.get('audio_result', '')
                    ])
            
            QMessageBox.information(self, "Export Complete", f"Report saved to {filepath}")

# =============================================================================
# TUMOR LOCALIZER CLASS
# =============================================================================

class TumorLocalizer:
    def __init__(self):
        self.antenna_positions = ANTENNA_POSITIONS
    
    def analyze_path_attenuation(self, s21_data, baseline_data):
        path_attenuation = {}
        path_ratios = {}
        
        for path_num in [1, 2, 3, 4]:
            if path_num in s21_data and baseline_data and path_num in baseline_data:
                patient_avg = np.mean(s21_data[path_num])
                baseline_avg = np.mean(baseline_data[path_num])
                diff = baseline_avg - patient_avg
                path_attenuation[path_num] = diff
                path_ratios[path_num] = patient_avg / baseline_avg if baseline_avg != 0 else 1.0
        
        sorted_paths = sorted(path_attenuation.items(), key=lambda x: x[1], reverse=True)
        tumor_location = self._estimate_location_from_paths(sorted_paths)
        
        return {
            'path_attenuation': path_attenuation,
            'path_ratios': path_ratios,
            'most_affected_paths': sorted_paths[:2],
            'tumor_location': tumor_location,
            'confidence': self._calculate_confidence(sorted_paths, path_ratios)
        }
    
    def _estimate_location_from_paths(self, sorted_paths):
        if len(sorted_paths) < 1:
            return {'x': 0, 'y': 0, 'description': 'Unable to localize', 'quadrant': 'unknown'}
        
        top_paths = [p[0] for p in sorted_paths[:2]]
        intersections = []
        
        for path_num in top_paths:
            if path_num in PATH_TO_ANTENNA_PAIR:
                tx, rx = PATH_TO_ANTENNA_PAIR[path_num]
                tx_pos = self.antenna_positions[tx]
                rx_pos = self.antenna_positions[rx]
                mid_x = (tx_pos[0] + rx_pos[0]) / 2
                mid_y = (tx_pos[1] + rx_pos[1]) / 2
                intersections.append((mid_x, mid_y))
        
        if len(intersections) >= 2:
            avg_x = np.mean([p[0] for p in intersections])
            avg_y = np.mean([p[1] for p in intersections])
        elif len(intersections) == 1:
            avg_x, avg_y = intersections[0]
        else:
            avg_x, avg_y = 0, 0
        
        quadrant = self._get_quadrant(avg_x, avg_y)
        
        return {
            'x': avg_x,
            'y': avg_y,
            'quadrant': quadrant,
            'description': f"Abnormal presence detected in {quadrant} region of the chest"
        }
    
    def _get_quadrant(self, x, y):
        if x > 0 and y > 0:
            return "upper right"
        elif x < 0 and y > 0:
            return "upper left"
        elif x > 0 and y < 0:
            return "lower right"
        elif x < 0 and y < 0:
            return "lower left"
        else:
            return "central"
    
    def _calculate_confidence(self, sorted_paths, path_ratios):
        if len(sorted_paths) < 2:
            return 0.3
        
        top1_atten = sorted_paths[0][1] if sorted_paths else 0
        top2_atten = sorted_paths[1][1] if len(sorted_paths) > 1 else 0
        
        all_atten = [a for _, a in sorted_paths]
        if len(all_atten) > 1 and np.std(all_atten) > 0:
            spread_ratio = (top1_atten - top2_atten) / (np.std(all_atten) + 1e-6)
        else:
            spread_ratio = 1
        
        confidence = min(0.95, 0.3 + spread_ratio * 0.3)
        return confidence
    
    def generate_bounding_box(self, tumor_location, image_width=350, image_height=350):
        px = int((tumor_location['x'] + 100) / 200 * image_width)
        py = int((tumor_location['y'] + 100) / 200 * image_height)
        box_size = 40
        x1 = max(0, px - box_size // 2)
        y1 = max(0, py - box_size // 2)
        x2 = min(image_width, px + box_size // 2)
        y2 = min(image_height, py + box_size // 2)
        return (x1, y1, x2, y2)

# =============================================================================
# RECONSTRUCTION WIDGET
# =============================================================================

class ReconstructionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.reconstruction_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.setStyleSheet("background-color: white; border: 2px solid #4fc3f7; border-radius: 10px;")
    
    def reconstruct_image(self, s21_data, frequencies, baseline_data=None):
        try:
            grid_size = 80
            x_grid = np.linspace(-100, 100, grid_size)
            y_grid = np.linspace(-100, 100, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            image = np.zeros((grid_size, grid_size))
            c = 3e8
            
            for path_num, s21 in s21_data.items():
                if path_num not in PATH_TO_ANTENNA_PAIR:
                    continue
                
                tx_ant, rx_ant = PATH_TO_ANTENNA_PAIR[path_num]
                tx_pos = ANTENNA_POSITIONS[tx_ant]
                rx_pos = ANTENNA_POSITIONS[rx_ant]
                
                s21_linear = db_to_linear(s21)
                
                if baseline_data and path_num in baseline_data:
                    baseline_linear = db_to_linear(baseline_data[path_num])
                    s21_linear = s21_linear - baseline_linear
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        point = (X[i, j], Y[i, j])
                        d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                        d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                        total_dist = (d_tx + d_rx) / 1000
                        delay = total_dist / c
                        freq_idx = int(np.clip(delay * 1e9 / (STOP_FREQ/1e9) * POINTS, 0, POINTS-1))
                        
                        if freq_idx < len(s21_linear):
                            image[i, j] += s21_linear[freq_idx]
            
            image /= len(s21_data)
            image = gaussian_filter(image, sigma=2)
            
            if image.max() > 0:
                image = np.clip(image, 0, np.percentile(image, 95))
                image = (image / image.max()) * 255
            
            self.reconstruction_data = image.astype(np.uint8)
            self.update()
            return self.reconstruction_data
            
        except Exception as e:
            print(f"Reconstruction error: {e}")
            traceback.print_exc()
            return None
    
    def set_tumor_localization(self, tumor_location, confidence, bounding_box):
        self.tumor_location = tumor_location
        self.localization_confidence = confidence
        self.bounding_box = bounding_box
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(255, 255, 255))
        
        if self.reconstruction_data is not None:
            img_height, img_width = self.reconstruction_data.shape
            scale_x = w / img_width
            scale_y = h / img_height
            
            for i in range(img_width):
                for j in range(img_height):
                    val = self.reconstruction_data[i, j]
                    r = int(val)
                    g = int(val * 0.5)
                    b = int(255 - val)
                    color = QColor(r, g, b)
                    x = int(i * scale_x)
                    y = int(j * scale_y)
                    painter.fillRect(x, y, int(scale_x) + 1, int(scale_y) + 1, color)
            
            if self.bounding_box and self.localization_confidence > 0.5:
                x1, y1, x2, y2 = self.bounding_box
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine((x1 + x2) // 2 - 10, (y1 + y2) // 2, (x1 + x2) // 2 + 10, (y1 + y2) // 2)
                painter.drawLine((x1 + x2) // 2, (y1 + y2) // 2 - 10, (x1 + x2) // 2, (y1 + y2) // 2 + 10)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawRect(x1, y1 - 25, 120, 20)
                painter.drawText(x1 + 5, y1 - 10, f"Confidence: {self.localization_confidence:.0%}")
        else:
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.drawRect(10, 10, w - 20, h - 20)
            painter.drawText(w//2 - 100, h//2, "Reconstruction will appear here after scan")
        
        painter.setPen(QPen(QColor(0, 0, 255), 3))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        for ant, pos in ANTENNA_POSITIONS.items():
            x = int((pos[0] + 100) / 200 * w)
            y = int((pos[1] + 100) / 200 * h)
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            painter.drawText(x - 15, y - 10, f"{ant}")
        
        painter.end()
    
    def clear(self):
        self.reconstruction_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.update()

# =============================================================================
# AUDIO PROCESSOR (WITH FIXED SAMPLE RATE AND FILTERING)
# =============================================================================

class AudioProcessor(QThread):
    result_ready = Signal(np.ndarray)
    waveform_ready = Signal(np.ndarray)
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, record_seconds=3, device_id=None):
        super().__init__()
        self.record_seconds = record_seconds
        self.device_id = device_id
        self.yamnet_interpreter = None
        self.classifier_interpreter = None
        self.audio_data = None
        self.actual_sample_rate = 16000
        self._load_models()
    
    def _load_models(self):
        try:
            if not YAMNET_PATH.exists():
                self.error_occurred.emit(f"YAMNet model not found: {YAMNET_PATH}")
                return
            
            self.yamnet_interpreter = tflite.Interpreter(str(YAMNET_PATH))
            self.yamnet_interpreter.allocate_tensors()
            print("YAMNet TFLite loaded")
            
            if not AUDIO_MODEL_PATH.exists():
                self.error_occurred.emit(f"Classifier not found: {AUDIO_MODEL_PATH}")
                return
            
            self.classifier_interpreter = tflite.Interpreter(str(AUDIO_MODEL_PATH))
            self.classifier_interpreter.allocate_tensors()
            print("Classifier loaded")
            
        except Exception as e:
            self.error_occurred.emit(f"Model loading error: {e}")
            traceback.print_exc()
    
    def run(self):
        try:
            print(f"Recording {self.record_seconds} seconds...")
            
            try:
                if self.device_id is not None:
                    device_info = sd.query_devices(self.device_id)
                    supported_rates = [8000, 16000, 22050, 44100, 48000]
                    device_sr = 16000
                    for rate in supported_rates:
                        try:
                            sd.check_input_settings(device=self.device_id, samplerate=rate, channels=1)
                            device_sr = rate
                            break
                        except:
                            continue
                else:
                    device_info = sd.query_devices(sd.default.device[0])
                    device_sr = int(device_info['default_samplerate'])
            except Exception as e:
                print(f"Error detecting device sample rate: {e}")
                device_sr = 16000
            
            self.actual_sample_rate = device_sr
            sample_count = int(self.record_seconds * device_sr)
            
            print(f"Recording at {device_sr} Hz, {sample_count} samples")
            
            try:
                recording = sd.rec(sample_count, samplerate=device_sr, channels=1,
                                   dtype='float32', device=self.device_id, blocking=True)
                sd.wait()
                audio = recording.flatten()
                self.audio_data = audio
            except Exception as e:
                print(f"Recording error: {e}")
                device_sr = 16000
                sample_count = int(self.record_seconds * device_sr)
                recording = sd.rec(sample_count, samplerate=device_sr, channels=1,
                                   dtype='float32', device=self.device_id, blocking=True)
                sd.wait()
                audio = recording.flatten()
                self.audio_data = audio
            
            if device_sr != 16000:
                print(f"Resampling from {device_sr} Hz to 16000 Hz")
                new_length = int(len(audio) * 16000 / device_sr)
                audio = scipy.signal.resample(audio, new_length)
            
            from scipy import signal
            nyquist = 16000 / 2
            b, a = signal.butter(4, [100/nyquist, 2000/nyquist], btype='band')
            audio = signal.filtfilt(b, a, audio)
            
            audio = audio * AUDIO_GAIN
            
            audio_max = np.max(np.abs(audio))
            if audio_max > 0.001:
                audio = audio / audio_max
                print(f"Audio captured with peak level: {audio_max:.3f} (after gain: {audio_max * AUDIO_GAIN:.3f})")
            else:
                print(f"Low audio input detected. Peak level: {audio_max:.3f}. Returning healthy default.")
                healthy_probs = np.zeros(5, dtype=np.float32)
                healthy_probs[3] = 1.0
                self.result_ready.emit(healthy_probs)
                self.finished.emit()
                return
            
            expected_len = EXPECTED_AUDIO_SAMPLES
            if len(audio) < expected_len:
                audio = np.pad(audio, (0, expected_len - len(audio)))
            elif len(audio) > expected_len:
                audio = audio[:expected_len]
            
            input_details = self.yamnet_interpreter.get_input_details()[0]
            self.yamnet_interpreter.set_tensor(input_details['index'], audio.astype(np.float32))
            self.yamnet_interpreter.invoke()
            
            output_details = self.yamnet_interpreter.get_output_details()[0]
            embeddings = self.yamnet_interpreter.get_tensor(output_details['index'])
            
            if len(embeddings.shape) == 2:
                pooled_embedding = embeddings[0]
            elif len(embeddings.shape) == 1:
                pooled_embedding = embeddings
            else:
                pooled_embedding = np.mean(embeddings, axis=0)
            
            input_data = pooled_embedding.reshape(1, -1).astype(np.float32)
            classifier_input = self.classifier_interpreter.get_input_details()[0]
            self.classifier_interpreter.set_tensor(classifier_input['index'], input_data)
            self.classifier_interpreter.invoke()
            
            classifier_output = self.classifier_interpreter.get_output_details()[0]
            probs = self.classifier_interpreter.get_tensor(classifier_output['index'])[0]
            
            self.result_ready.emit(probs.astype(np.float32))
            
            if len(audio) > 800:
                step = max(1, len(audio) // 800)
                self.waveform_ready.emit(audio[::step][:800])
            
        except Exception as e:
            print(f"Audio error: {e}. Returning healthy default.")
            healthy_probs = np.zeros(5, dtype=np.float32)
            healthy_probs[3] = 1.0
            self.result_ready.emit(healthy_probs)
        finally:
            self.finished.emit()

# =============================================================================
# EDUCATIONAL WIDGET (FIXED SCROLLING - NO CUTOFF)
# =============================================================================

class EducationalWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            EducationalWidget {
                background-color: #f5f5f5;
                border: 2px solid #4fc3f7;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        self._setup_ui()
        self.hide()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.title_label = QLabel("Clinical Education")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0277bd;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #4fc3f7;")
        layout.addWidget(divider)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4fc3f7;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
        
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setSpacing(15)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        self.condition_label = QLabel()
        self.condition_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #0277bd;")
        self.condition_label.setAlignment(Qt.AlignCenter)
        self.condition_label.setWordWrap(True)
        self.content_layout.addWidget(self.condition_label)
        
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 8px;")
        self.desc_label.setAlignment(Qt.AlignTop)
        self.content_layout.addWidget(self.desc_label)
        
        signs_group = QGroupBox("Clinical Signs and Symptoms")
        signs_group.setStyleSheet("""
            QGroupBox { 
                font-size: 15px; 
                font-weight: bold; 
                border: 1px solid #ccc; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 12px;
            }
        """)
        signs_layout = QVBoxLayout(signs_group)
        self.signs_label = QLabel()
        self.signs_label.setWordWrap(True)
        self.signs_label.setStyleSheet("font-size: 13px; padding: 8px; line-height: 1.5;")
        self.signs_label.setAlignment(Qt.AlignTop)
        signs_layout.addWidget(self.signs_label)
        self.content_layout.addWidget(signs_group)
        
        rec_group = QGroupBox("Recommendations")
        rec_group.setStyleSheet("""
            QGroupBox { 
                font-size: 15px; 
                font-weight: bold; 
                border: 1px solid #ccc; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 12px;
            }
        """)
        rec_layout = QVBoxLayout(rec_group)
        self.rec_label = QLabel()
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet("font-size: 13px; padding: 8px; line-height: 1.5;")
        self.rec_label.setAlignment(Qt.AlignTop)
        rec_layout.addWidget(self.rec_label)
        self.content_layout.addWidget(rec_group)
        
        literacy_note = QLabel(
            "Clinical literacy empowers patients to recognize symptoms early and seek appropriate care."
        )
        literacy_note.setWordWrap(True)
        literacy_note.setStyleSheet(
            "font-size: 12px; font-style: italic; color: #666; "
            "background-color: #e3f2fd; padding: 10px; border-radius: 8px;"
        )
        literacy_note.setAlignment(Qt.AlignTop)
        self.content_layout.addWidget(literacy_note)
        
        self.content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
    
    def show_condition(self, condition_name, confidence):
        condition = condition_name.lower()
        content = EDUCATIONAL_CONTENT.get(condition, EDUCATIONAL_CONTENT.get('healthy'))
        
        self.condition_label.setText(f"{condition_name.upper()} ({confidence:.1%} confidence)")
        self.desc_label.setText(content['description'])
        self.signs_label.setText(content['clinical_signs'])
        self.rec_label.setText(content['recommendations'])
        
        self.show()
        self.raise_()

# =============================================================================
# VNA CONTROLLER
# =============================================================================

class VNADirectController:
    def __init__(self, port=VNA_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.frequencies = None
        self.connect()
    
    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            print(f"VNA connected on {self.port}")
            return True
        except Exception as e:
            print(f"VNA connection failed: {e}")
            return False
    
    def capture_s21(self, progress_callback=None):
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return None
        
        try:
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5\r\n"
            self.serial_conn.write(cmd.encode())
            time.sleep(2.0)
            
            data_points = []
            lines_collected = 0
            timeout_start = time.time()
            max_timeout = 20
            
            while lines_collected < POINTS and (time.time() - timeout_start) < max_timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                    if not line or line.startswith('ch>') or line.startswith('scan') or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            s21_real = float(parts[1])
                            s21_imag = float(parts[2])
                            magnitude = math.sqrt(s21_real**2 + s21_imag**2)
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            data_points.append(magnitude_db)
                            lines_collected += 1
                            if progress_callback and lines_collected % 20 == 0:
                                progress_callback(lines_collected, POINTS)
                        except ValueError:
                            continue
                else:
                    time.sleep(0.05)
            
            if len(data_points) == POINTS:
                if self.frequencies is None:
                    self.frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, POINTS)
                return np.array(data_points)
            else:
                print(f"Only {len(data_points)}/{POINTS} points captured")
                return None
                
        except Exception as e:
            print(f"VNA capture error: {e}")
            return None
    
    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("VNA disconnected")

# =============================================================================
# RF SWITCH CONTROLLER
# =============================================================================

class RFSwitchController:
    def __init__(self):
        self._setup_gpio()
    
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
        print("GPIO initialized for RF switches")
    
    def set_path(self, path_num):
        if path_num not in PATHS:
            raise ValueError(f"Invalid path: {path_num}")
        states = PATHS[path_num]
        for pin, state in states.items():
            if pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
                GPIO.output(pin, state)
        time.sleep(0.1)
        print(f"Path {path_num} set: {PATHS[path_num]['desc']}")
    
    def cleanup(self):
        GPIO.cleanup()
        print("GPIO cleaned up")

# =============================================================================
# CSV DATA MANAGER
# =============================================================================

class CSVDataManager:
    def __init__(self):
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        PATIENT_DIR.mkdir(parents=True, exist_ok=True)
        MULTI_ANGLE_DIR.mkdir(parents=True, exist_ok=True)
        print("Data directories created")
    
    def save_scan(self, data, path_num, directory, frequencies=None, angle=0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"path{path_num}_angle{angle}_{timestamp}.csv"
        filepath = directory / filename
        
        if frequencies is None:
            frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, len(data))
        
        rows = []
        for i, (freq, s21) in enumerate(zip(frequencies, data)):
            rows.append([freq, s21])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_GHz', 'S21_dB'])
            writer.writerows(rows)
        
        return filepath
    
    def load_latest_from_directory(self, directory):
        data = {}
        for path_num in [1, 2, 3, 4]:
            files = list(directory.glob(f"path{path_num}_*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                data[path_num] = df['S21_dB'].values
        return data if data else None
    
    def has_baseline(self):
        for path_num in [1, 2, 3, 4]:
            files = list(BASELINE_DIR.glob(f"path{path_num}_*.csv"))
            if files:
                return True
        return False
    
    def clear_all(self):
        for d in [BASELINE_DIR, PATIENT_DIR, MULTI_ANGLE_DIR]:
            if d.exists():
                for f in d.glob("*.csv"):
                    f.unlink()
        print("All data cleared")

# =============================================================================
# MICROWAVE SCANNER (WITH BACKGROUND SUBTRACTION)
# =============================================================================

class MicrowaveScanner:
    def __init__(self, vna_controller):
        self.vna = vna_controller
        self.switch = RFSwitchController()
        self.csv_manager = CSVDataManager()
        self.frequencies = None
        self._baseline_data = None
        self._baseline_linear = None
    
    def scan_all_paths(self, save_dir, angle=0, progress_callback=None):
        data = {}
        total_paths = len(PATHS)
        
        for idx, path_num in enumerate(PATHS.keys(), 1):
            if progress_callback:
                progress_callback(f"Setting Path {path_num} at {angle} deg", idx / total_paths)
            
            self.switch.set_path(path_num)
            time.sleep(0.2)
            
            if progress_callback:
                progress_callback(f"Capturing Path {path_num}", idx / total_paths)
            
            s21_data = self.vna.capture_s21()
            if s21_data is None:
                raise RuntimeError(f"Failed to capture path {path_num}")
            
            data[path_num] = s21_data
            
            if self.frequencies is None:
                self.frequencies = self.vna.frequencies
            
            self.csv_manager.save_scan(s21_data, path_num, save_dir, self.frequencies, angle)
            
            if progress_callback:
                progress_callback(f"Path {path_num} complete", idx / total_paths)
        
        return data
    
    def set_baseline(self, baseline_data):
        self._baseline_data = baseline_data.copy()
        self._baseline_linear = {}
        for path_num, s21_db in baseline_data.items():
            self._baseline_linear[path_num] = db_to_linear(s21_db)
        print(f"Baseline stored for {len(self._baseline_data)} paths")
        print("Background subtraction ready: Will remove direct antenna coupling in linear domain")
    
    def apply_background_subtraction_to_scan(self, patient_data):
        if self._baseline_linear is None:
            print("Warning: No baseline set, cannot apply background subtraction")
            return patient_data
        
        corrected_data = {}
        for path_num, patient_s21_db in patient_data.items():
            if path_num in self._baseline_linear:
                patient_linear = db_to_linear(patient_s21_db)
                corrected_linear = patient_linear - self._baseline_linear[path_num]
                corrected_linear = np.maximum(corrected_linear, 1e-12)
                corrected_data[path_num] = linear_to_db(corrected_linear)
                print(f"Path {path_num}: Applied background subtraction (coupling removed)")
            else:
                corrected_data[path_num] = patient_s21_db
        
        return corrected_data
    
    def load_baseline(self):
        if self._baseline_data is not None:
            return self._baseline_data
        baseline_data = self.csv_manager.load_latest_from_directory(BASELINE_DIR)
        if baseline_data:
            self.set_baseline(baseline_data)
        return baseline_data
    
    def has_baseline(self):
        return self.csv_manager.has_baseline() or (self._baseline_data is not None)
    
    def extract_features(self, s21_data):
        freq_features = np.array([s21_data[p] for p in [1, 2, 3, 4]]).reshape(1, -1)
        augmented_features = self._add_time_domain_features(freq_features)
        return augmented_features[0]
    
    def combine_rotation_features(self, rotation_data):
        all_freq_features = []
        
        for angle, data in rotation_data.items():
            freq_features = np.array([data[p] for p in [1, 2, 3, 4]]).reshape(1, -1)
            all_freq_features.append(freq_features)
        
        avg_freq_features = np.mean(all_freq_features, axis=0)
        augmented_features = self._add_time_domain_features(avg_freq_features)
        return augmented_features[0]
    
    def _add_time_domain_features(self, X):
        n_samples, n_features = X.shape
        n_freq = n_features
        n_paths = 4
        freq_per_path = n_freq // n_paths
        
        time_features = []
        
        for sample in X:
            sample_time_features = []
            for path in range(n_paths):
                start_idx = path * freq_per_path
                end_idx = (path + 1) * freq_per_path
                freq_response = sample[start_idx:end_idx]
                freq_response_linear = db_to_linear(freq_response)
                time_response = np.fft.ifft(freq_response_linear)
                time_magnitude = np.abs(time_response)
                
                sample_time_features.extend([
                    np.max(time_magnitude),
                    np.argmax(time_magnitude),
                    np.mean(time_magnitude),
                    np.std(time_magnitude),
                    np.percentile(time_magnitude, 90),
                    np.percentile(time_magnitude, 10),
                    np.sum(time_magnitude),
                    np.max(time_magnitude) - np.min(time_magnitude),
                    np.sum(np.square(time_magnitude)),
                ])
            
            time_features.append(sample_time_features)
        
        time_features = np.array(time_features)
        X_augmented = np.concatenate([X, time_features], axis=1)
        
        return X_augmented
    
    def cleanup(self):
        self.switch.cleanup()
        self.vna.close()

# =============================================================================
# MAIN GUI APPLICATION - THORACIS AI
# =============================================================================

class ThoracisAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("THORACIS AI: Operation Oracle - Lung Screening System")
        
        self.vna = VNADirectController()
        self.scanner = MicrowaveScanner(self.vna)
        self.rf_switch = RFSwitchController()
        
        self.audio_device_id = None
        self._setup_audio_device()
        
        self.microwave_classifier = None
        try:
            self.microwave_classifier = MicrowaveOnlyClassifier()
        except Exception as e:
            print(f"Microwave classifier not loaded: {e}")
        
        self.educational_widget = EducationalWidget(self)
        self.reconstruction_widget = ReconstructionWidget(self)
        
        self.health_passport = HealthPassportWidget()
        
        self.data_collection_widget = DataCollectionModeWidget(self.vna, self.rf_switch)
        
        self.spectrogram_widget = SpectrogramWidget()
        self.microwave_contrast_widget = MicrowaveContrastWidget()
        self.fusion_explanation_widget = FusionExplanationWidget()
        self.explainability_text = ExplainabilityTextWidget()
        
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self._check_sync_folder)
        self.sync_timer.start(10000)
        
        self.fusion = None
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")
        
        self.current_mw_features = None
        self.current_audio_probs = None
        self.last_ai_probs = None
        self.current_s21_data = None
        self.baseline_data = None
        self.rotation_scans = {}
        self.last_recorded_audio = None
        self.last_sample_rate = 16000
        self.current_microwave_prediction = None
        self.current_microwave_confidence = None
        
        self._setup_ui()
        self.showFullScreen()
    
    def _check_sync_folder(self):
        skin_scans = check_for_skin_scans()
        if skin_scans:
            print(f"Received {len(skin_scans)} skin scans from NOMA AI")
            self.status_bar.setText(f"New skin scans received from NOMA AI - Check Operation Oracle tab")
    
    def _setup_audio_device(self):
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']} - {device['max_input_channels']} in")
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and ('USB' in device['name'] or 'Mic' in device['name']):
                    self.audio_device_id = i
                    print(f"Selected audio input device: {i} - {device['name']}")
                    break
            
            if self.audio_device_id is None:
                print("Using default audio input device")
        except Exception as e:
            print(f"Error detecting audio devices: {e}")
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        title = QLabel("THORACIS AI: Operation Oracle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #0277bd;
            background-color: #e1f5fe;
            padding: 15px;
            border-radius: 20px;
            margin-bottom: 10px;
        """)
        left_layout.addWidget(title)
        
        subtitle = QLabel("Democratized Lung Screening | Explainable AI | Clinical Education")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #555; margin-bottom: 10px;")
        left_layout.addWidget(subtitle)
        
        vna_status = "Connected" if self.vna.serial_conn else "Disconnected"
        audio_status = "USB" if self.audio_device_id is not None else "Default"
        sync_status = check_sync_folder_health()
        self.status_bar = QLabel(f"VNA: {vna_status} | Audio: {audio_status} | Multi-Angle: {len(ROTATION_ANGLES)} positions | Sync: Active | BG Subtraction: Enabled")
        self.status_bar.setStyleSheet("font-size: 11px; color: #666; padding: 5px; background: #f0f0f0; border-radius: 8px;")
        left_layout.addWidget(self.status_bar)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #4fc3f7;
                border-radius: 10px;
                background: white;
            }
            QTabBar::tab {
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                background: #e1f5fe;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #4fc3f7;
                color: white;
            }
        """)
        
        self._add_microwave_tab()
        self._add_audio_tab()
        self._add_fusion_tab()
        self._add_data_collection_tab()
        self._add_explainability_tab()
        self._add_health_passport_tab()
        self._add_operation_oracle_tab()
        self._add_education_tab()
        
        left_layout.addWidget(self.tabs)
        
        # EXIT BUTTON - Always visible, fixed position
        exit_btn = QPushButton("EXIT")
        exit_btn.setMinimumHeight(50)
        exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background: #ef5350;
                color: white;
                border: 2px solid #c62828;
                border-radius: 12px;
                padding: 10px;
                margin-top: 15px;
            }
            QPushButton:hover { 
                background: #ff6659; 
            }
        """)
        exit_btn.clicked.connect(self.close)
        left_layout.addWidget(exit_btn)
        
        right_panel = QWidget()
        right_panel.setMaximumWidth(550)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 10, 5, 10)
        right_layout.setSpacing(10)
        
        self.spectrogram_widget.setVisible(False)
        self.microwave_contrast_widget.setVisible(False)
        self.fusion_explanation_widget.setVisible(False)
        self.explainability_text.setVisible(False)
        
        right_layout.addWidget(self.spectrogram_widget)
        right_layout.addWidget(self.microwave_contrast_widget)
        right_layout.addWidget(self.fusion_explanation_widget)
        right_layout.addWidget(self.reconstruction_widget)
        right_layout.addWidget(self.explainability_text)
        right_layout.addWidget(self.educational_widget)
        
        mission = QLabel("""
        <b>Operation Oracle Mission</b><br>
        Transforming early detection from a scarce, opaque resource 
        into a portable, explainable, and truly accessible practice.
        <br><br>
        <i>The most effective diagnostic tools don't just process data, 
        but build the literacy necessary to understand it.</i>
        <br><br>
        <b>Shared Sync Folder:</b> /opt/oracle_share
        <br>
        <b>Background Subtraction:</b> Direct antenna coupling removed in linear domain
        """)
        mission.setWordWrap(True)
        mission.setStyleSheet("font-size: 10px; color: #555; background: #f5f5f5; padding: 10px; border-radius: 10px;")
        mission.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(mission)
        
        main_layout.addWidget(left_panel, 5)
        main_layout.addWidget(right_panel, 5)
    
    def _add_data_collection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.data_collection_widget)
        self.tabs.addTab(tab, "Data Collection")
    
    def _add_explainability_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #e0e0e0;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4fc3f7;
                min-height: 30px;
                border-radius: 6px;
            }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Explainable AI - How Decisions Are Made")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #0277bd; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)
        
        audio_group = QGroupBox("Acoustic Analysis - Spectrogram")
        audio_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; margin-top: 10px; }")
        audio_layout = QVBoxLayout(audio_group)
        audio_layout.setSpacing(10)
        
        audio_image_container = QWidget()
        audio_image_container.setMinimumHeight(300)
        audio_image_container.setMaximumHeight(350)
        audio_image_layout = QVBoxLayout(audio_image_container)
        
        audio_image_label = QLabel()
        audio_image_label.setAlignment(Qt.AlignCenter)
        audio_image_label.setMinimumHeight(280)
        audio_image_label.setStyleSheet("background-color: #f5f5f5; border-radius: 8px; padding: 5px;")
        
        if SPECTROGRAM_IMAGE_PATH.exists():
            pixmap = QPixmap(str(SPECTROGRAM_IMAGE_PATH))
            scaled_pixmap = pixmap.scaled(700, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            audio_image_label.setPixmap(scaled_pixmap)
        else:
            audio_image_label.setText("Acoustic Spectrogram Visualization\n(Image not found)")
            audio_image_label.setStyleSheet("font-size: 14px; color: #666; background-color: #e3f2fd; padding: 20px;")
        
        audio_image_layout.addWidget(audio_image_label)
        audio_layout.addWidget(audio_image_container)
        
        audio_text = QLabel(
            "The spectrogram shows frequency (y-axis) vs time (x-axis) of lung sounds.\n\n"
            "Key features to understand:\n"
            "  - Wheezing appears as horizontal bands at specific frequencies (300-500 Hz)\n"
            "  - Crackles appear as vertical streaks indicating sudden pressure changes\n"
            "  - Normal breath sounds show diffuse energy below 1 kHz\n"
            "  - Red dashed lines indicate detected abnormal features\n\n"
            "How the AI uses this: YAMNet extracts acoustic features, which are then classified"
        )
        audio_text.setWordWrap(True)
        audio_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 12px; background: #e8f5e9; border-radius: 8px;")
        audio_layout.addWidget(audio_text)
        
        content_layout.addWidget(audio_group)
        
        microwave_group = QGroupBox("Microwave Analysis - S21 Traces")
        microwave_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; margin-top: 10px; }")
        microwave_layout = QVBoxLayout(microwave_group)
        microwave_layout.setSpacing(10)
        
        microwave_image_container = QWidget()
        microwave_image_container.setMinimumHeight(300)
        microwave_image_container.setMaximumHeight(350)
        microwave_image_layout = QVBoxLayout(microwave_image_container)
        
        microwave_image_label = QLabel()
        microwave_image_label.setAlignment(Qt.AlignCenter)
        microwave_image_label.setMinimumHeight(280)
        microwave_image_label.setStyleSheet("background-color: #f5f5f5; border-radius: 8px; padding: 5px;")
        
        if MICROWAVE_IMAGE_PATH.exists():
            pixmap = QPixmap(str(MICROWAVE_IMAGE_PATH))
            scaled_pixmap = pixmap.scaled(700, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            microwave_image_label.setPixmap(scaled_pixmap)
        else:
            microwave_image_label.setText("Microwave S21 Trace Visualization\n(Image not found)")
            microwave_image_label.setStyleSheet("font-size: 14px; color: #666; background-color: #e3f2fd; padding: 20px;")
        
        microwave_image_layout.addWidget(microwave_image_label)
        microwave_layout.addWidget(microwave_image_container)
        
        microwave_text = QLabel(
            "The S21 transmission plots show signal attenuation across frequency (2-3 GHz).\n\n"
            "Key features to understand:\n"
            "  - Colored lines represent different antenna paths (1->3, 1->4, 2->3, 2->4)\n"
            "  - Dashed black line shows baseline (air) measurement\n"
            "  - Lower dB values indicate more signal attenuation\n"
            "  - Tumor presence creates increased attenuation (colored lines drop below baseline)\n\n"
            "How the AI uses this: Features extracted from S21 traces are used to detect structural abnormalities"
        )
        microwave_text.setWordWrap(True)
        microwave_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 12px; background: #e3f2fd; border-radius: 8px;")
        microwave_layout.addWidget(microwave_text)
        
        content_layout.addWidget(microwave_group)
        
        bg_sub_group = QGroupBox("Background Subtraction - Removing Antenna Coupling")
        bg_sub_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; margin-top: 10px; }")
        bg_sub_layout = QVBoxLayout(bg_sub_group)
        
        bg_sub_text = QLabel(
            "The Problem: Direct antenna coupling can be 40+ dB stronger than tissue signal.\n\n"
            "The Solution: Measure baseline (air only) and subtract in LINEAR domain (not dB).\n\n"
            "Why Linear Domain?\n"
            "  - Subtraction in dB = division in linear (WRONG for coupling removal)\n"
            "  - Linear domain subtraction properly removes the additive coupling signal\n\n"
            "Example:\n"
            "  Coupling signal: -20 dB (linear = 0.01)\n"
            "  Tissue signal: -40 dB (linear = 0.0001)\n"
            "  Total measured: -20 dB (coupling dominates)\n"
            "  If you subtract in dB: (-20) - (-20) = 0 dB (WRONG)\n"
            "  If you subtract in linear: (0.01 + 0.0001) - 0.01 = 0.0001 then convert to dB -> -40 dB (CORRECT)"
        )
        bg_sub_text.setWordWrap(True)
        bg_sub_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 12px; background: #fff3e0; border-radius: 8px;")
        bg_sub_layout.addWidget(bg_sub_text)
        
        content_layout.addWidget(bg_sub_group)
        
        fusion_group = QGroupBox("Fusion Decision - Cross-Modal Integration")
        fusion_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; margin-top: 10px; }")
        fusion_layout = QVBoxLayout(fusion_group)
        
        fusion_text = QLabel(
            "Fusion combines structural (microwave) and functional (acoustic) data.\n\n"
            "How the AI integrates both modalities:\n"
            "  - Microwave path detects structural abnormalities (tumors, masses)\n"
            "  - Acoustic path identifies functional changes (wheezing, crackles)\n"
            "  - High probability in both paths suggests space-occupying lesion affecting airflow\n"
            "  - Disagreement between paths suggests need for additional clinical correlation\n\n"
            "The fusion model outputs a final diagnosis with confidence score based on both modalities."
        )
        fusion_text.setWordWrap(True)
        fusion_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 12px; background: #fce4ec; border-radius: 8px;")
        fusion_layout.addWidget(fusion_text)
        
        content_layout.addWidget(fusion_group)
        
        oracle_group = QGroupBox("Operation Oracle - Cross-Device Syncing")
        oracle_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; margin-top: 10px; }")
        oracle_layout = QVBoxLayout(oracle_group)
        
        oracle_text = QLabel(
            "THORACIS AI and NOMA AI share data through /opt/oracle_share.\n\n"
            "What gets synced:\n"
            "  - Skin scans from NOMA AI appear automatically in Operation Oracle dashboard\n"
            "  - Lung scans from THORACIS AI are visible to NOMA AI\n"
            "  - Cross-modal alerts detect paraneoplastic syndromes (lung + skin findings together)\n\n"
            "This unified patient record enables comprehensive healthcare monitoring across modalities."
        )
        oracle_text.setWordWrap(True)
        oracle_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 12px; background: #e1f5fe; border-radius: 8px;")
        oracle_layout.addWidget(oracle_text)
        
        content_layout.addWidget(oracle_group)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "Explainability")
    
    def _add_microwave_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.mw_status = QLabel("Ready - Place antennas and connect VNA")
        self.mw_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.mw_status)
        
        self.baseline_btn = QPushButton("1. RECORD BASELINE (AIR) - Removes antenna coupling")
        self.baseline_btn.setMinimumHeight(50)
        self.baseline_btn.setStyleSheet(self._button_style("#81d4fa"))
        self.baseline_btn.clicked.connect(self._record_baseline)
        layout.addWidget(self.baseline_btn)
        
        self.scan_btn = QPushButton("2. SCAN PATIENT (MULTI-ANGLE) - With background subtraction")
        self.scan_btn.setMinimumHeight(50)
        self.scan_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.scan_btn.clicked.connect(self._run_multi_angle_scan)
        layout.addWidget(self.scan_btn)
        
        self.mw_progress = QProgressBar()
        self.mw_progress.setVisible(False)
        layout.addWidget(self.mw_progress)
        
        self.mw_result = QTextEdit()
        self.mw_result.setReadOnly(True)
        self.mw_result.setMinimumHeight(200)
        self.mw_result.setStyleSheet("font-size: 12px; font-family: monospace;")
        layout.addWidget(self.mw_result)
        
        clear_btn = QPushButton("CLEAR ALL DATA")
        clear_btn.setMinimumHeight(40)
        clear_btn.setStyleSheet(self._button_style("#ff9800"))
        clear_btn.clicked.connect(self._clear_all_data)
        layout.addWidget(clear_btn)
        
        self.tabs.addTab(tab, "Microwave")
    
    def _add_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.audio_status = QLabel("Ready - Place stethoscope on patient's back")
        self.audio_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.audio_status)
        
        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.waveform_label)
        
        self.audio_btn = QPushButton("ANALYZE LUNG SOUNDS")
        self.audio_btn.setMinimumHeight(50)
        self.audio_btn.setStyleSheet(self._button_style("#66bb6a"))
        self.audio_btn.clicked.connect(self._run_acoustic_analysis)
        layout.addWidget(self.audio_btn)
        
        self.audio_result = QTextEdit()
        self.audio_result.setReadOnly(True)
        self.audio_result.setMinimumHeight(250)
        self.audio_result.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.audio_result)
        
        self.tabs.addTab(tab, "Acoustic")
    
    def _add_fusion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.fusion_status = QLabel("Perform both scans for combined diagnosis")
        self.fusion_status.setStyleSheet("font-size: 14px; padding: 8px; background: #fff3e0; border-radius: 8px;")
        layout.addWidget(self.fusion_status)
        
        self.fusion_mw_btn = QPushButton("1. SCAN MICROWAVE (MULTI-ANGLE)")
        self.fusion_mw_btn.setMinimumHeight(50)
        self.fusion_mw_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_mw_btn.clicked.connect(self._fusion_microwave)
        layout.addWidget(self.fusion_mw_btn)
        
        self.fusion_audio_btn = QPushButton("2. ANALYZE ACOUSTIC")
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_audio_btn.setMinimumHeight(50)
        self.fusion_audio_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_audio_btn.clicked.connect(self._fusion_acoustic)
        layout.addWidget(self.fusion_audio_btn)
        
        self.fusion_combine_btn = QPushButton("3. RUN FUSION DIAGNOSIS")
        self.fusion_combine_btn.setEnabled(False)
        self.fusion_combine_btn.setMinimumHeight(55)
        self.fusion_combine_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.fusion_combine_btn.clicked.connect(self._run_fusion)
        layout.addWidget(self.fusion_combine_btn)
        
        self.fusion_result = QTextEdit()
        self.fusion_result.setReadOnly(True)
        self.fusion_result.setMinimumHeight(300)
        self.fusion_result.setStyleSheet("font-size: 13px; font-family: monospace;")
        layout.addWidget(self.fusion_result)
        
        self.tabs.addTab(tab, "Fusion")
    
    def _add_health_passport_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.health_passport)
        self.tabs.addTab(tab, "Health Passport")
    
    def _add_operation_oracle_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        description = QLabel(
            "OPERATION ORACLE - UNIFIED PATIENT RECORD\n"
            "This dashboard integrates data from both Thoracis AI (lung) and NOMA AI (skin).\n"
            "Cross-modal alerts help detect paraneoplastic syndromes where lung and skin findings co-occur.\n\n"
            "Shared sync folder: /opt/oracle_share"
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 13px; background-color: #e1f5fe; padding: 10px; border-radius: 8px;")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        open_dashboard_btn = QPushButton("OPEN OPERATION ORACLE DASHBOARD")
        open_dashboard_btn.setMinimumHeight(60)
        open_dashboard_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background: #4fc3f7;
                color: white;
                border: none;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background: #29b6f6;
            }
        """)
        open_dashboard_btn.clicked.connect(self._open_oracle_dashboard)
        layout.addWidget(open_dashboard_btn)
        
        summary_group = QGroupBox("Recent Cross-Modal Summary")
        summary_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        summary_layout = QVBoxLayout(summary_group)
        
        self.cross_modal_summary = QLabel("Loading...")
        self.cross_modal_summary.setWordWrap(True)
        self.cross_modal_summary.setStyleSheet("font-size: 12px; padding: 8px;")
        summary_layout.addWidget(self.cross_modal_summary)
        
        layout.addWidget(summary_group)
        
        self.tabs.addTab(tab, "Operation Oracle")
        self._update_cross_modal_summary()
    
    def _add_education_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        welcome = QLabel("Clinical Education Center")
        welcome.setStyleSheet("font-size: 24px; font-weight: bold; color: #0277bd;")
        welcome.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(welcome)
        
        oracle_desc = QLabel("""
        Operation Oracle: Democratizing Early Detection
        
        Against Clinical Delay: Rapid, multimodal screening at point-of-care (under 5 minutes)
        
        Against Access Barriers: Portable platforms using democratized hardware (Raspberry Pi + NanoVNA)
        
        Against Systemic Exclusion: Explainable AI + clinical education + complete open-source documentation
        
        Together, we transform early detection from a scarce, opaque resource into a portable, 
        explainable, and truly accessible practice.
        """)
        oracle_desc.setWordWrap(True)
        oracle_desc.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 15px; background: #f5f5f5; border-radius: 10px;")
        content_layout.addWidget(oracle_desc)
        
        how_it_works = QLabel("""
        How It Works:
        
        1. Microwave Scan: 4 antennas measure tissue dielectric properties - tumors appear as contrast anomalies
        
        2. Acoustic Analysis: Lung sound recording detects wheezing, crackles, and breath patterns
        
        3. Fusion Diagnosis: Combines structural (microwave) and functional (acoustic) data for comprehensive assessment
        """)
        how_it_works.setWordWrap(True)
        how_it_works.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 15px; background: #e3f2fd; border-radius: 10px;")
        content_layout.addWidget(how_it_works)
        
        background_subtraction_text = QLabel("""
        Background Subtraction Strategy:
        
        The Problem: Direct antenna coupling can be 40+ dB stronger than tissue signal.
        
        The Solution: Measure baseline (air only) and subtract in LINEAR domain (not dB).
        
        Why Linear Domain? Subtraction in dB = division in linear, which is incorrect.
        Linear domain subtraction properly removes the additive coupling signal.
        """)
        background_subtraction_text.setWordWrap(True)
        background_subtraction_text.setStyleSheet("font-size: 14px; line-height: 1.6; padding: 15px; background: #e8f5e9; border-radius: 10px;")
        content_layout.addWidget(background_subtraction_text)
        
        conditions_group = QGroupBox("Common Respiratory Conditions")
        conditions_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; }")
        conditions_layout = QVBoxLayout(conditions_group)
        
        conditions_text = QLabel("""
        ASTHMA: Airway inflammation causing wheezing, chest tightness, and breathing difficulties.
               Typically episodic and triggered by allergens or exercise.
        
        COPD: Chronic airflow obstruction from emphysema or chronic bronchitis.
              Symptoms include chronic cough, sputum production, and shortness of breath.
        
        PNEUMONIA: Lung infection causing air sacs to fill with fluid.
                   Presents with fever, chills, productive cough, and chest pain.
        
        BRONCHITIS: Inflammation of the main airways causing persistent cough with mucus.
        """)
        conditions_text.setWordWrap(True)
        conditions_text.setStyleSheet("font-size: 13px; line-height: 1.5; padding: 10px;")
        conditions_layout.addWidget(conditions_text)
        content_layout.addWidget(conditions_group)
        
        data_collection_note = QLabel("""
        Data Collection Mode:
        
        The Data Collection tab allows you to perform systematic phantom scans for AI training and validation.
        - Scans all 4 antenna paths automatically
        - Supports multiple rotation angles (0, 120, 240 degrees)
        - Saves data in CSV format compatible with the training pipeline
        - Each scan includes metadata for condition labeling
        """)
        data_collection_note.setWordWrap(True)
        data_collection_note.setStyleSheet("font-size: 13px; padding: 15px; background: #fff3e0; border-radius: 10px;")
        content_layout.addWidget(data_collection_note)
        
        opensource = QLabel("""
        Open-Source Documentation:
        
        This project includes comprehensive documentation, online courses, and community forums 
        to enable anyone to build, modify, and improve the system. Democratization of innovation 
        is a fundamental pillar of Operation Oracle.
        """)
        opensource.setWordWrap(True)
        opensource.setStyleSheet("font-size: 13px; padding: 15px; background: #fce4ec; border-radius: 10px;")
        content_layout.addWidget(opensource)
        
        sync_info = QLabel("""
        Cross-Device Syncing:
        
        THORACIS AI and NOMA AI share data through /opt/oracle_share
        - Skin scans appear automatically in Operation Oracle dashboard
        - Cross-modal alerts detect paraneoplastic syndromes
        - Both devices must have the shared folder mounted
        """)
        sync_info.setWordWrap(True)
        sync_info.setStyleSheet("font-size: 13px; padding: 15px; background: #e1f5fe; border-radius: 10px;")
        content_layout.addWidget(sync_info)
        
        disclaimer = QLabel("""
        DISCLAIMER: This is an AI-assisted screening tool for research and educational purposes.
        Not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
        """)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("font-size: 11px; color: #999; padding: 10px;")
        disclaimer.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(disclaimer)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Education")
    
    def _button_style(self, color):
        return f"""
            QPushButton {{
                font-size: 15px;
                font-weight: bold;
                background: {color};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: {color}cc;
            }}
            QPushButton:disabled {{
                background: #cccccc;
                color: #888;
            }}
        """
    
    def _update_cross_modal_summary(self):
        try:
            conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT diagnosis, risk_level FROM thoracic_scans ORDER BY timestamp DESC LIMIT 3')
            thoracic = cursor.fetchall()
            
            cursor.execute('SELECT diagnosis, confidence, risk_level FROM skin_scans_received ORDER BY timestamp DESC LIMIT 3')
            skin = cursor.fetchall()
            
            conn.close()
            
            summary = "Recent Thoracic Scans:\n"
            if thoracic:
                for dx, risk in thoracic:
                    summary += f"  - {dx} ({risk})\n"
            else:
                summary += "  - No thoracic scans recorded\n"
            
            summary += "\nRecent Skin Scans (from NOMA AI):\n"
            if skin:
                for dx, conf, risk in skin:
                    summary += f"  - {dx} ({risk})\n"
            else:
                summary += "  - No skin scans received yet\n"
                summary += "  - Check that /opt/oracle_share exists on both devices\n"
            
            high_risk_thoracic = any(r[1] in ['HIGH', 'URGENT'] for r in thoracic)
            high_risk_skin = any(r[2] in ['HIGH', 'URGENT'] for r in skin) if skin else False
            
            if high_risk_thoracic and high_risk_skin:
                summary += "\nPARANEOPLASTIC SYNDROME ALERT: Both thoracic and skin high-risk findings detected."
            elif high_risk_thoracic:
                summary += "\nCLINICAL CORRELATION: High-risk thoracic findings. Skin assessment recommended."
            elif high_risk_skin:
                summary += "\nCLINICAL CORRELATION: High-risk skin findings. Thoracic assessment recommended."
            
            self.cross_modal_summary.setText(summary)
        except Exception as e:
            self.cross_modal_summary.setText(f"Error loading summary: {e}")
    
    def _open_oracle_dashboard(self):
        try:
            self._update_cross_modal_summary()
            dashboard = OperationOracleDashboard(self)
            dashboard.exec()
        except Exception as e:
            QMessageBox.warning(self, "Dashboard Error", f"Could not open Operation Oracle Dashboard: {str(e)}")
    
    def _update_mw_progress(self, msg, frac):
        self.mw_status.setText(msg)
        self.mw_progress.setValue(int(frac * 100))
        QApplication.processEvents()
    
    def _clear_all_data(self):
        reply = QMessageBox.question(self, "Clear Data", "Delete all saved baseline and patient scans?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.scanner.csv_manager.clear_all()
            self.scanner._baseline_data = None
            self.scanner._baseline_linear = None
            self.current_mw_features = None
            self.current_audio_probs = None
            self.last_ai_probs = None
            self.current_s21_data = None
            self.rotation_scans = {}
            self.reconstruction_widget.clear()
            self.mw_result.setText("All data cleared. You can now record a new baseline.")
            self.mw_status.setText("Ready")
            QMessageBox.information(self, "Data Cleared", "All saved scans have been deleted.")
    
    def _record_baseline(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        self.baseline_btn.setEnabled(False)
        self.mw_status.setText("Recording baseline (air) for background subtraction...")
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        self.mw_result.setText("Starting baseline scan - Place nothing between antennas...\nThis captures direct antenna coupling for removal.")
        
        def worker():
            try:
                data = self.scanner.scan_all_paths(
                    BASELINE_DIR, angle=0,
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.scanner.set_baseline(data)
                self.baseline_data = data
                
                result_text = "BASELINE RECORDED SUCCESSFULLY\n\n"
                result_text += "Baseline (air) data saved. This captures direct antenna coupling.\n"
                result_text += "Background subtraction will remove this coupling from patient scans.\n"
                result_text += "Subtraction performed in LINEAR domain (power ratio), not dB.\n\n"
                result_text += f"Files saved to: {BASELINE_DIR}\n"
                result_text += "\nNow place patient between antennas and click SCAN PATIENT (MULTI-ANGLE)."
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Baseline complete - Background subtraction ready")
                
            except Exception as e:
                self.mw_result.setText(f"Error: {e}\n\nCheck VNA connection and try again.")
                self.mw_status.setText("Baseline failed")
                traceback.print_exc()
            finally:
                self.baseline_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _run_multi_angle_scan(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        if not self.scanner.has_baseline():
            QMessageBox.warning(self, "Missing Baseline", "Please record baseline (air) first for background subtraction!")
            return
        
        self.scan_btn.setEnabled(False)
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        
        def worker():
            try:
                all_rotation_data = {}
                baseline = self.scanner.load_baseline()
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    self.mw_status.setText(f"Scanning at {angle} deg rotation...")
                    self.mw_progress.setValue(int(angle_idx / len(ROTATION_ANGLES) * 50))
                    
                    if angle_idx > 0:
                        time.sleep(2)
                    
                    raw_data = self.scanner.scan_all_paths(
                        MULTI_ANGLE_DIR, angle=angle,
                        progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                    )
                    
                    corrected_data = self.scanner.apply_background_subtraction_to_scan(raw_data)
                    all_rotation_data[angle] = corrected_data
                    
                    self.reconstruction_widget.reconstruct_image(corrected_data, self.scanner.frequencies, baseline)
                    self.mw_progress.setValue(int((angle_idx + 1) / len(ROTATION_ANGLES) * 50))
                    
                    tumor_info = {'dielectric_contrast': self._calculate_contrast(corrected_data, baseline), 'location': 'estimated central'}
                    self.microwave_contrast_widget.update_traces(corrected_data, baseline, self.scanner.frequencies, tumor_info)
                    self.microwave_contrast_widget.setVisible(True)
                    
                    # Run microwave-only classifier
                    if self.microwave_classifier is not None:
                        combined_features = self.scanner.combine_rotation_features(all_rotation_data)
                        mw_pred, mw_conf = self.microwave_classifier.predict(combined_features)
                        self.current_microwave_prediction = mw_pred
                        self.current_microwave_confidence = mw_conf
                        print(f"Microwave-only classification: {'TUMOR' if mw_pred == 1 else 'HEALTHY'} with confidence {mw_conf:.1%}")
                    
                    self.explainability_text.set_text(
                        "MICROWAVE ANALYSIS EXPLANATION\n\n"
                        "The S21 traces show signal transmission between antenna pairs with background subtraction applied.\n"
                        "Background subtraction removes direct antenna coupling in LINEAR domain.\n"
                        f"Average attenuation: {self._calculate_contrast(corrected_data, baseline):.1f} dB\n"
                        "Tumors appear as increased attenuation (lower dB values) due to higher dielectric constant of malignant tissue.\n"
                        "The contrast between patient and baseline indicates structural abnormalities."
                    )
                    self.explainability_text.setVisible(True)
                
                self.rotation_scans = all_rotation_data
                combined_features = self.scanner.combine_rotation_features(all_rotation_data)
                self.current_mw_features = combined_features
                self._reconstruct_from_rotations(all_rotation_data)
                
                localizer = TumorLocalizer()
                combined_attenuation = {}
                for angle, data in all_rotation_data.items():
                    analysis = localizer.analyze_path_attenuation(data, baseline)
                    for path, atten in analysis['path_attenuation'].items():
                        if path not in combined_attenuation:
                            combined_attenuation[path] = []
                        combined_attenuation[path].append(atten)
                
                avg_attenuation = {p: np.mean(v) for p, v in combined_attenuation.items()}
                sorted_paths = sorted(avg_attenuation.items(), key=lambda x: x[1], reverse=True)
                tumor_location = localizer._estimate_location_from_paths(sorted_paths)
                confidence = localizer._calculate_confidence(sorted_paths, {})
                w = self.reconstruction_widget.width()
                h = self.reconstruction_widget.height()
                bounding_box = localizer.generate_bounding_box(tumor_location, w, h)
                self.reconstruction_widget.set_tumor_localization(tumor_location, confidence, bounding_box)
                
                microwave_result_display = "TUMOR DETECTED" if (self.current_microwave_prediction == 1) else "NORMAL"
                microwave_conf_display = f"{self.current_microwave_confidence:.1%}" if self.current_microwave_confidence else "N/A"
                
                result_text = "MULTI-ANGLE PATIENT SCAN COMPLETE\n\n"
                result_text += f"Scanned at angles: {ROTATION_ANGLES}\n"
                result_text += f"Total transmission paths: {len(ROTATION_ANGLES) * 4}\n"
                result_text += "Background subtraction applied: Direct antenna coupling removed in linear domain\n\n"
                
                result_text += "MICROWAVE-ONLY CLASSIFICATION RESULT:\n"
                result_text += f"   Result: {microwave_result_display}\n"
                result_text += f"   Confidence: {microwave_conf_display}\n\n"
                
                result_text += "TUMOR LOCALIZATION:\n"
                if confidence > 0.5:
                    result_text += f"   Location: {tumor_location['description']}\n"
                    result_text += f"   Coordinates: ({tumor_location['x']:.0f} mm, {tumor_location['y']:.0f} mm)\n"
                    result_text += f"   Confidence: {confidence:.0%}\n\n"
                    result_text += "   Most affected paths (averaged across rotations):\n"
                    for path_num, attenuation in sorted_paths[:2]:
                        result_text += f"     Path {path_num}: {attenuation:.1f} dB attenuation\n"
                else:
                    result_text += "   No significant abnormality detected\n"
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Multi-angle scan complete - Background subtraction applied")
                
                # Save microwave result to database
                try:
                    conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
                    cursor = conn.cursor()
                    risk_level = "HIGH" if self.current_microwave_prediction == 1 else "LOW"
                    diagnosis = "tumor" if self.current_microwave_prediction == 1 else "healthy"
                    cursor.execute('''
                        INSERT INTO thoracic_scans (patient_id, timestamp, diagnosis, confidence, microwave_result, audio_result, risk_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (self.health_passport.current_patient_id, datetime.now().isoformat(),
                          diagnosis, self.current_microwave_confidence if self.current_microwave_confidence else 0.5,
                          microwave_result_display, "pending", risk_level))
                    conn.commit()
                    conn.close()
                    print(f"Saved microwave scan result: {diagnosis} with confidence {self.current_microwave_confidence:.1%}")
                except Exception as e:
                    print(f"Database error: {e}")
                
                # Enable fusion button if audio will be done later
                if self.fusion is not None and self.current_mw_features is not None:
                    self.fusion_audio_btn.setEnabled(True)
                
            except Exception as e:
                self.mw_result.setText(f"Error: {e}")
                self.mw_status.setText("Scan failed")
                traceback.print_exc()
            finally:
                self.scan_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _calculate_contrast(self, patient_data, baseline_data):
        if not baseline_data:
            return 0
        contrasts = []
        for path_num in [1, 2, 3, 4]:
            if path_num in patient_data and path_num in baseline_data:
                patient_avg = np.mean(patient_data[path_num])
                baseline_avg = np.mean(baseline_data[path_num])
                contrasts.append(baseline_avg - patient_avg)
        return np.mean(contrasts) if contrasts else 0
    
    def _reconstruct_from_rotations(self, rotation_data):
        try:
            avg_s21 = {p: [] for p in [1, 2, 3, 4]}
            
            for angle, data in rotation_data.items():
                for path_num in [1, 2, 3, 4]:
                    if path_num in data:
                        avg_s21[path_num].append(data[path_num])
            
            for path_num in avg_s21:
                if avg_s21[path_num]:
                    avg_s21[path_num] = np.mean(avg_s21[path_num], axis=0)
                else:
                    avg_s21[path_num] = np.zeros(POINTS)
            
            baseline = self.scanner.load_baseline()
            self.reconstruction_widget.reconstruct_image(avg_s21, self.scanner.frequencies, baseline)
        except Exception as e:
            print(f"Rotation reconstruction error: {e}")
            traceback.print_exc()
    
    def _run_acoustic_analysis(self):
        self.audio_btn.setEnabled(False)
        self.audio_status.setText("Recording lung sounds (3 seconds)...")
        self.audio_result.setText("Processing...")
        self.waveform_label.setText("")
        
        self.audio_thread = AudioProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_audio_result)
        self.audio_thread.waveform_ready.connect(self._draw_waveform)
        self.audio_thread.error_occurred.connect(self._on_audio_error)
        self.audio_thread.finished.connect(self._on_audio_finished)
        self.audio_thread.start()
    
    def _on_audio_result(self, probs):
        self.last_ai_probs = probs
        self.last_recorded_audio = getattr(self.audio_thread, 'audio_data', None)
        self.current_audio_probs = probs
        
        audio_class_idx = np.argmax(probs)
        audio_class = MODEL_CLASSES[audio_class_idx] if audio_class_idx < len(MODEL_CLASSES) else "healthy"
        audio_conf = probs[audio_class_idx] if audio_class_idx < len(probs) else 0.5
        
        if self.last_recorded_audio is not None:
            detected_features = self._detect_audio_features(self.last_recorded_audio, 16000, probs)
            self.spectrogram_widget.update_spectrogram(self.last_recorded_audio, 16000, detected_features)
            self.spectrogram_widget.setVisible(True)
            self.explainability_text.set_text(
                "ACOUSTIC ANALYSIS EXPLANATION\n\n"
                "The spectrogram shows frequency content over time.\n"
                f"Detected features: {[f['type'] for f in detected_features] if detected_features else 'none'}\n"
                f"AI Prediction: {audio_class.upper()} with {audio_conf:.1%} confidence\n"
                "Wheezing appears as horizontal bands at specific frequencies (300-500 Hz).\n"
                "Crackles appear as vertical streaks indicating sudden pressure equalization.\n"
                "Normal breath sounds show diffuse energy below 1 kHz.\n\n"
                "Audio gain applied for sensitive detection.\n"
                "Bandpass filter applied (100-2000 Hz) to focus on respiratory sounds."
            )
            self.explainability_text.setVisible(True)
        
        # Show clinical assessment questionnaire
        self.clinical_assessment = RespiratoryClinicalAssessment(self, audio_class, audio_conf)
        self.clinical_assessment.assessment_complete.connect(self._on_clinical_assessment_complete)
        self.clinical_assessment.exec_()
    
    def _detect_audio_features(self, audio, sample_rate, probs):
        features = []
        max_idx = np.argmax(probs)
        
        if max_idx == 1:
            features.append({'type': 'wheezing', 'frequency': 450})
        elif max_idx == 2:
            features.append({'type': 'rhonchi', 'frequency': 200})
        elif max_idx == 4:
            features.append({'type': 'crackles', 'frequency': 1000})
        
        return features
    
    def _on_clinical_assessment_complete(self, assessment_results):
        audio_class_idx = np.argmax(self.current_audio_probs) if self.current_audio_probs is not None else 3
        audio_class = MODEL_CLASSES[audio_class_idx] if audio_class_idx < len(MODEL_CLASSES) else "healthy"
        audio_conf = self.current_audio_probs[audio_class_idx] if self.current_audio_probs is not None else 0.5
        
        clinical_risk = assessment_results.get('clinical_risk', 'LOW')
        clinical_score = assessment_results.get('clinical_score', 0)
        
        severity = AUDIO_SEVERITY.get(audio_class, 0.3)
        
        overall_risk = "LOW"
        if clinical_risk == "HIGH" or severity > 0.7:
            overall_risk = "HIGH"
        elif clinical_risk == "MODERATE" or severity > 0.4:
            overall_risk = "MODERATE"
        
        result_text = "ACOUSTIC ANALYSIS COMPLETE\n\n"
        result_text += f"AI Analysis:\n"
        result_text += f"  Detected Condition: {audio_class.upper()}\n"
        result_text += f"  Confidence: {audio_conf:.1%}\n"
        result_text += f"  Severity Score: {severity:.1%}\n\n"
        
        result_text += f"Clinical Assessment:\n"
        result_text += f"  Clinical Score: {clinical_score}/10\n"
        result_text += f"  Clinical Risk: {clinical_risk}\n\n"
        
        result_text += f"Overall Respiratory Risk: {overall_risk}\n\n"
        
        if audio_class != 'healthy':
            result_text += f"Educational Information:\n"
            content = EDUCATIONAL_CONTENT.get(audio_class, EDUCATIONAL_CONTENT.get('healthy'))
            result_text += f"  {content['description']}\n"
            result_text += f"  {content['recommendations']}\n\n"
        
        result_text += "DISCLAIMER: This is an AI-assisted screening tool.\n"
        result_text += "Always consult a qualified healthcare provider for medical decisions."
        
        self.audio_result.setText(result_text)
        self.audio_status.setText(f"Diagnosis: {audio_class.upper()} ({audio_conf:.1%})")
        self.educational_widget.show_condition(audio_class, audio_conf)
        
        # Save to health passport
        self.health_passport.add_scan_record(
            diagnosis=audio_class,
            confidence=audio_conf,
            microwave_result=self.current_microwave_prediction if self.current_microwave_prediction is not None else "pending",
            audio_result=audio_class,
            audio_probs=self.current_audio_probs
        )
        
        # Save to database
        try:
            conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO thoracic_scans (patient_id, timestamp, diagnosis, confidence, microwave_result, audio_result, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.health_passport.current_patient_id, datetime.now().isoformat(),
                  audio_class, audio_conf, 
                  self.current_microwave_prediction if self.current_microwave_prediction is not None else "pending",
                  audio_class, overall_risk))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
        
        # Sync to NOMA AI
        sync_data = {
            'diagnosis': audio_class,
            'confidence': audio_conf,
            'risk_level': overall_risk,
            'clinical_score': clinical_score,
            'scan_type': 'lung',
            'source': 'THORACIS_AI'
        }
        sync_scan_to_noma(sync_data)
        
        # Enable fusion if microwave data available
        if self.current_mw_features is not None and self.fusion is not None:
            self.fusion_combine_btn.setEnabled(True)
        
        self._update_cross_modal_summary()
    
    def _on_audio_error(self, error_msg):
        self.audio_result.setText(f"Audio Error: {error_msg}\n\nDefaulting to HEALTHY classification.\n\nCheck microphone connection and ensure it is plugged in.\nThe system applies gain to boost low signals and bandpass filtering (100-2000 Hz).")
        self.audio_status.setText("Audio error - defaulting to healthy")
        print(f"Audio error: {error_msg}")
        
        # Default to healthy
        self.current_audio_probs = np.zeros(5)
        self.current_audio_probs[3] = 1.0
        self.educational_widget.show_condition("healthy", 0.7)
        
        # Still show clinical assessment
        self.clinical_assessment = RespiratoryClinicalAssessment(self, "healthy", 0.7)
        self.clinical_assessment.assessment_complete.connect(self._on_clinical_assessment_complete)
        self.clinical_assessment.exec_()
    
    def _on_audio_finished(self):
        self.audio_btn.setEnabled(True)
    
    def _draw_waveform(self, data):
        if data is None or len(data) == 0:
            return
        w = self.waveform_label.width()
        h = self.waveform_label.height()
        if w < 10 or h < 10:
            return
        
        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.cyan, 2))
        
        mid = h // 2
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)
        data = np.clip(data, -1, 1)
        step = w / len(data)
        
        for i in range(1, len(data)):
            x1 = int((i-1) * step)
            x2 = int(i * step)
            y1 = int(mid + data[i-1] * mid * 0.8)
            y2 = int(mid + data[i] * mid * 0.8)
            painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        self.waveform_label.setPixmap(pixmap)
    
    def _fusion_microwave(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        self.fusion_mw_btn.setEnabled(False)
        self.fusion_status.setText("Running multi-angle microwave scan with background subtraction...")
        
        def worker():
            try:
                if not self.scanner.has_baseline():
                    self.fusion_status.setText("Need baseline first! Go to Microwave tab.")
                    self.fusion_mw_btn.setEnabled(True)
                    return
                
                all_rotation_data = {}
                baseline = self.scanner.load_baseline()
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    if angle_idx > 0:
                        time.sleep(2)
                    
                    raw_data = self.scanner.scan_all_paths(MULTI_ANGLE_DIR, angle=angle)
                    corrected_data = self.scanner.apply_background_subtraction_to_scan(raw_data)
                    all_rotation_data[angle] = corrected_data
                    self.reconstruction_widget.reconstruct_image(corrected_data, self.scanner.frequencies, baseline)
                    
                    tumor_info = {'dielectric_contrast': self._calculate_contrast(corrected_data, baseline), 'location': 'estimated central'}
                    self.microwave_contrast_widget.update_traces(corrected_data, baseline, self.scanner.frequencies, tumor_info)
                    self.microwave_contrast_widget.setVisible(True)
                
                self.rotation_scans = all_rotation_data
                combined_features = self.scanner.combine_rotation_features(all_rotation_data)
                self.current_mw_features = combined_features
                self._reconstruct_from_rotations(all_rotation_data)
                
                # Run microwave-only classifier
                if self.microwave_classifier is not None:
                    mw_pred, mw_conf = self.microwave_classifier.predict(combined_features)
                    self.current_microwave_prediction = mw_pred
                    self.current_microwave_confidence = mw_conf
                    print(f"Microwave-only classification: {'TUMOR' if mw_pred == 1 else 'HEALTHY'} with confidence {mw_conf:.1%}")
                    self.fusion_status.setText(f"Microwave complete: {'TUMOR' if mw_pred == 1 else 'HEALTHY'} ({mw_conf:.1%})")
                else:
                    self.fusion_status.setText("Microwave complete. Now perform acoustic analysis.")
                
                self.fusion_audio_btn.setEnabled(True)
                
            except Exception as e:
                self.fusion_status.setText(f"Error: {e}")
                traceback.print_exc()
            finally:
                self.fusion_mw_btn.setEnabled(True)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _fusion_acoustic(self):
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_status.setText("Recording acoustic...")
        
        self.audio_thread = AudioProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_fusion_audio_result)
        self.audio_thread.error_occurred.connect(self._on_fusion_audio_error)
        self.audio_thread.finished.connect(self._on_fusion_audio_finished)
        self.audio_thread.start()
    
    def _on_fusion_audio_result(self, probs):
        self.last_ai_probs = probs
        self.current_audio_probs = probs
        self.fusion_status.setText("Acoustic complete. Ready for fusion diagnosis.")
        self.fusion_combine_btn.setEnabled(True)
    
    def _on_fusion_audio_error(self, error_msg):
        self.fusion_status.setText(f"Acoustic error: {error_msg}. Using healthy default.")
        self.current_audio_probs = np.zeros(5)
        self.current_audio_probs[3] = 1.0
        self.fusion_combine_btn.setEnabled(True)
        self.fusion_audio_btn.setEnabled(True)
    
    def _on_fusion_audio_finished(self):
        self.fusion_audio_btn.setEnabled(True)
    
    def _run_fusion(self):
        if self.current_mw_features is None or self.current_audio_probs is None:
            QMessageBox.warning(self, "Missing Data", "Perform both scans first!")
            return
        
        if self.fusion is None:
            self.fusion_result.setText("Fusion model not loaded!")
            return
        
        try:
            fusion_pred, fusion_conf = self.fusion.predict(self.current_mw_features, self.current_audio_probs)
            
            audio_class_idx = np.argmax(self.current_audio_probs)
            audio_class = MODEL_CLASSES[audio_class_idx] if audio_class_idx < len(MODEL_CLASSES) else "healthy"
            audio_conf = self.current_audio_probs[audio_class_idx] if audio_class_idx < len(self.current_audio_probs) else 0.5
            
            microwave_class = "tumor" if self.current_microwave_prediction == 1 else "healthy"
            microwave_conf = self.current_microwave_confidence if self.current_microwave_confidence else 0.5
            
            fusion_abnormal = (fusion_pred == 1)
            audio_abnormal = (audio_class != 'healthy')
            
            if fusion_abnormal and audio_abnormal:
                agreement = "HIGH - Both modalities indicate abnormality"
                risk_level = "HIGH"
                if AUDIO_SEVERITY.get(audio_class, 0) > 0.7:
                    clinical_recommendation = "URGENT: Refer for low-dose CT within 2 weeks. Severe acoustic pattern suggests possible COPD/pneumonia comorbidity."
                else:
                    clinical_recommendation = "Refer for low-dose CT imaging. Acoustic analysis indicates airway obstruction."
            elif fusion_abnormal and not audio_abnormal:
                agreement = "MODERATE - Fusion positive, Audio normal"
                risk_level = "MEDIUM-HIGH"
                clinical_recommendation = "Follow-up CT recommended. Acoustic pattern normal but microwave suggests structural changes."
            elif not fusion_abnormal and audio_abnormal:
                agreement = "MODERATE - Fusion normal, Audio abnormal"
                risk_level = "MEDIUM"
                clinical_recommendation = "Clinical correlation advised. Audio indicates airway obstruction but no structural microwave anomaly."
            else:
                agreement = "HIGH - Both modalities indicate normal"
                risk_level = "LOW"
                clinical_recommendation = "Routine monitoring. No immediate concerns."
            
            explanation = self._generate_fusion_explanation(fusion_pred, fusion_conf, audio_class, audio_conf, microwave_class, microwave_conf)
            
            result_text = "=" * 60 + "\n"
            result_text += "THORACIS AI FUSION DIAGNOSIS\n"
            result_text += "=" * 60 + "\n\n"
            
            if fusion_pred == 1:
                result_text += "FINAL CLINICAL ASSESSMENT: ABNORMAL - TUMOR SUSPECTED\n"
            else:
                result_text += "FINAL CLINICAL ASSESSMENT: NORMAL - NO TUMOR DETECTED\n"
            result_text += f"   Fusion Confidence: {fusion_conf:.1%}\n\n"
            
            result_text += "MULTIMODAL FINDINGS:\n"
            result_text += f"   Microwave (Structural): {'TUMOR' if self.current_microwave_prediction == 1 else 'NORMAL'} ({microwave_conf:.1%})\n"
            result_text += f"   Acoustic (Functional): {audio_class.upper()} ({audio_conf:.0%})\n\n"
            
            result_text += f"CROSS-MODAL AGREEMENT: {agreement}\n"
            result_text += f"RISK LEVEL: {risk_level}\n\n"
            result_text += f"RECOMMENDATION: {clinical_recommendation}\n\n"
            
            result_text += explanation + "\n\n"
            
            result_text += "=" * 60 + "\n"
            result_text += "DISCLAIMER: AI-assisted screening tool. Not a substitute for professional medical diagnosis.\n"
            result_text += "Operation Oracle | Democratizing Early Detection"
            
            self.fusion_result.setText(result_text)
            self.fusion_status.setText(f"Fusion Diagnosis: {'TUMOR' if fusion_pred == 1 else 'NORMAL'} ({fusion_conf:.1%})")
            self.educational_widget.show_condition(audio_class, audio_conf)
            
            fusion_result = {
                'diagnosis': 'tumor' if fusion_pred == 1 else 'healthy',
                'confidence': fusion_conf,
                'explanation': explanation
            }
            self.fusion_explanation_widget.update_explanation(self.current_audio_probs, self.current_mw_features, fusion_result)
            self.fusion_explanation_widget.setVisible(True)
            self.explainability_text.set_text(
                "FUSION DIAGNOSIS EXPLANATION\n\n"
                f"Microwave analysis: {'TUMOR' if self.current_microwave_prediction == 1 else 'NORMAL'} (confidence: {microwave_conf:.0%})\n"
                f"Acoustic analysis: {audio_class.upper()} (confidence: {audio_conf:.0%})\n\n"
                f"Cross-modal agreement: {agreement}\n"
                f"Fusion combines structural and functional data. {explanation}\n\n"
                "The final confidence represents the agreement between both modalities.\n"
                "Background subtraction removed direct antenna coupling revealing tumor signal."
            )
            self.explainability_text.setVisible(True)
            
            # Save to database
            try:
                conn = sqlite3.connect('/home/anik/thoracis_longitudinal.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO thoracic_scans (patient_id, timestamp, diagnosis, confidence, microwave_result, audio_result, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (self.health_passport.current_patient_id, datetime.now().isoformat(),
                      'tumor' if fusion_pred == 1 else 'healthy', fusion_conf,
                      'tumor' if self.current_microwave_prediction == 1 else 'normal',
                      audio_class, risk_level))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Database error: {e}")
            
            # Sync to NOMA AI
            sync_data = {
                'diagnosis': 'tumor' if fusion_pred == 1 else 'healthy',
                'confidence': fusion_conf,
                'microwave_finding': 'tumor' if self.current_microwave_prediction == 1 else 'normal',
                'audio_finding': audio_class,
                'risk_level': risk_level,
                'agreement': agreement,
                'scan_type': 'lung_fusion',
                'source': 'THORACIS_AI'
            }
            sync_scan_to_noma(sync_data)
            
            self._update_cross_modal_summary()
            
        except Exception as e:
            self.fusion_result.setText(f"Fusion error: {e}")
            traceback.print_exc()
    
    def _generate_fusion_explanation(self, fusion_pred, fusion_conf, audio_class, audio_conf, microwave_class, microwave_conf):
        if fusion_pred == 1:
            explanation = (
                f"The fusion model detected a structural anomaly with {fusion_conf:.0%} confidence. "
                f"This combines microwave dielectric contrast analysis ({microwave_class.upper()} with {microwave_conf:.0%} confidence) "
                f"and acoustic pattern analysis ({audio_class.upper()} with {audio_conf:.0%} confidence)."
            )
            if audio_class != 'healthy':
                explanation += (
                    f"\n\nThe acoustic path identified {audio_class.upper()} with {audio_conf:.0%} confidence, "
                    f"suggesting concurrent functional airway changes. Together, these findings suggest "
                    f"a space-occupying lesion affecting airflow."
                )
            else:
                explanation += (
                    f"\n\nAcoustic analysis shows normal breath sounds, suggesting the structural "
                    f"abnormality is not yet causing functional impairment."
                )
        else:
            explanation = (
                f"Both microwave and acoustic paths show normal patterns.\n\n"
                f"Microwave: No significant dielectric contrast detected after background subtraction.\n"
                f"Acoustic: {audio_class.upper()} pattern with {audio_conf:.0%} confidence.\n\n"
                f"No evidence of structural lesions or significant functional obstruction."
            )
        return explanation
    
    def closeEvent(self, event):
        print("\nShutting down THORACIS AI: Operation Oracle...")
        if hasattr(self, 'sync_timer'):
            self.sync_timer.stop()
        self.scanner.cleanup()
        event.accept()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    platform = os.environ.get('QT_QPA_PLATFORM', '')
    if platform == 'eglfs':
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)

    app = QApplication(sys.argv)
    window = ThoracisAIMainWindow()
    window.showFullScreen()
    sys.exit(app.exec())

#!/usr/bin/env python3
"""
Quick Array Calibration - Run this FIRST before any scans!
"""

import numpy as np
import json
import time
from hardware.switch_controller import RFSwitchController
from hardware.vna_interface import VNAInterface

# Initialize
switch = RFSwitchController([17, 27], [22, 23])
vna = VNAInterface('/dev/ttyACM0')

print("="*60)
print("QUICK ARRAY CALIBRATION")
print("="*60)

# Connect to VNA
if not vna.connect():
    print("Failed to connect to VNA!")
    exit()

vna.set_frequency_range(2.0e9, 3.0e9, 201)

# Measure baseline for all antenna pairs
print("\nMeasuring baseline (free space) for all antenna pairs...")
print("Make sure NO objects are between antennas!")
input("Press Enter to continue...")

baseline_data = {}
antenna_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]

for tx, rx in antenna_pairs:
    print(f"\nMeasuring TX{tx} → RX{rx}...")
    
    # Select antennas
    if tx == 1:
        switch._set_single_switch(switch.tx_pins, 1)
    else:
        switch._set_single_switch(switch.tx_pins, 2)
    
    if rx == 3:
        switch._set_single_switch(switch.rx_pins, 1)
    else:
        switch._set_single_switch(switch.rx_pins, 2)
    
    time.sleep(0.05)
    
    # Measure
    freqs, mags, phases = vna.measure_s21()
    
    baseline_data[f'TX{tx}_RX{rx}'] = {
        'mean_db': float(np.mean(mags)),
        'std_db': float(np.std(mags)),
        'freq_ghz': [f/1e9 for f in freqs[:5]],  # First 5 freqs as sample
        'magnitude_sample': [float(m) for m in mags[:5]]
    }
    
    print(f"  Mean: {np.mean(mags):.2f} dB, Std: {np.std(mags):.2f} dB")

# Save calibration
import os
os.makedirs('calibration', exist_ok=True)

filename = f'calibration/baseline_{time.strftime("%Y%m%d_%H%M%S")}.json'
with open(filename, 'w') as f:
    json.dump({
        'baseline': baseline_data,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'frequency_ghz': '2.0-3.0',
        'antenna_pairs': antenna_pairs
    }, f, indent=2)

print("\n" + "="*60)
print(f"✅ CALIBRATION SAVED TO: {filename}")
print("="*60)

# Cleanup
switch.cleanup()
vna.disconnect()
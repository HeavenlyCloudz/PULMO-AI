import time
import csv
import numpy as np
from datetime import datetime
import os
from hardware.switch_controller import RFSwitchController
from hardware.vna_interface import VNAInterface

class ArrayScanner:
    """Main class for time-multiplexed array scanning."""
    
    def __init__(self, vna_port='/dev/ttyACM0', 
                 tx_pins=[17, 27], rx_pins=[22, 23],
                 start_freq=2.0e9, stop_freq=3.0e9, points=201):
        
        # Initialize hardware
        self.switch_controller = RFSwitchController(tx_pins, rx_pins)
        self.vna = VNAInterface(vna_port)
        
        # Scanning parameters
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.points = points
        
        # Data storage
        self.scan_data = []
        self.current_scan_id = None
        
        # Create data directory
        os.makedirs('data/datasets', exist_ok=True)
        os.makedirs('data/logs', exist_ok=True)
    
    def initialize(self):
        """Initialize hardware connections."""
        print("="*60)
        print("PULMO AI - MICROWAVE ARRAY SCANNER INITIALIZATION")
        print("="*60)
        
        # Connect to VNA
        if not self.vna.connect():
            raise ConnectionError("Failed to connect to VNA")
        
        # Set frequency range
        self.vna.set_frequency_range(self.start_freq, self.stop_freq, self.points)
        
        # Get all antenna pairs
        self.antenna_pairs = self.switch_controller.get_antenna_pairs()
        print(f"\nAntenna pairs to scan: {self.antenna_pairs}")
        print("Scanner initialized successfully!")
    
    def scan_single_pair(self, tx, rx, scan_label=""):
        """
        Scan a single TX-RX antenna pair.
        
        Returns: Dictionary with measurement data
        """
        print(f"\n[SCAN] TX{tx} → RX{rx} {scan_label}")
        
        # Select antenna pair
        self.switch_controller.select_antenna_pair(tx, rx)
        
        # Measure S21
        start_time = time.time()
        freqs, mags, phases = self.vna.measure_s21()
        scan_time = time.time() - start_time
        
        # Create data structure
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'tx_antenna': tx,
            'rx_antenna': rx,
            'scan_label': scan_label,
            'scan_time_seconds': scan_time,
            'frequencies_hz': freqs,
            's21_magnitudes_db': mags,
            's21_phases_deg': phases,
            'mean_magnitude_db': np.mean(mags),
            'std_magnitude_db': np.std(mags),
            'min_magnitude_db': np.min(mags),
            'max_magnitude_db': np.max(mags),
            'dynamic_range_db': np.max(mags) - np.min(mags)
        }
        
        print(f"  Mean S21: {measurement['mean_magnitude_db']:.2f} dB")
        print(f"  Dynamic range: {measurement['dynamic_range_db']:.2f} dB")
        print(f"  Scan time: {scan_time:.2f} seconds")
        
        return measurement
    
    def scan_full_array(self, scan_name="array_scan"):
        """
        Perform complete time-multiplexed scan of all antenna pairs.
        
        Returns: List of all measurements
        """
        print("\n" + "="*60)
        print(f"STARTING FULL ARRAY SCAN: {scan_name}")
        print("="*60)
        
        self.current_scan_id = f"{scan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_measurements = []
        
        start_total = time.time()
        
        # Scan each antenna pair
        for i, (tx, rx) in enumerate(self.antenna_pairs):
            measurement = self.scan_single_pair(tx, rx, f"pair_{i+1}")
            measurement['scan_id'] = self.current_scan_id
            all_measurements.append(measurement)
            
            # Progress update
            progress = ((i + 1) / len(self.antenna_pairs)) * 100
            print(f"Progress: {progress:.0f}% complete")
        
        total_time = time.time() - start_total
        print(f"\n✅ Full array scan completed in {total_time:.1f} seconds")
        print(f"   Collected {len(all_measurements)} antenna pair measurements")
        
        # Save data
        self.scan_data.extend(all_measurements)
        self._save_scan_data(all_measurements)
        
        return all_measurements
    
    def _save_scan_data(self, measurements):
        """Save scan data to CSV files."""
        # Save detailed data for each measurement
        for i, meas in enumerate(measurements):
            filename = f"data/datasets/{self.current_scan_id}_pair{i+1}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Frequency_Hz', 'S21_dB', 'Phase_deg'])
                
                for freq, mag, phase in zip(meas['frequencies_hz'], 
                                           meas['s21_magnitudes_db'], 
                                           meas['s21_phases_deg']):
                    writer.writerow([freq, mag, phase])
        
        # Save summary CSV
        summary_file = f"data/datasets/{self.current_scan_id}_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['tx', 'rx', 'mean_db', 'std_db', 'min_db', 'max_db', 
                         'dynamic_range_db', 'scan_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for meas in measurements:
                writer.writerow({
                    'tx': meas['tx_antenna'],
                    'rx': meas['rx_antenna'],
                    'mean_db': meas['mean_magnitude_db'],
                    'std_db': meas['std_magnitude_db'],
                    'min_db': meas['min_magnitude_db'],
                    'max_db': meas['max_magnitude_db'],
                    'dynamic_range_db': meas['dynamic_range_db'],
                    'scan_time': meas['scan_time_seconds']
                })
        
        # Log the scan
        log_file = f"data/logs/{self.current_scan_id}_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"Scan ID: {self.current_scan_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Frequency range: {self.start_freq/1e9:.2f} to {self.stop_freq/1e9:.2f} GHz\n")
            f.write(f"Points per scan: {self.points}\n")
            f.write(f"Antenna pairs: {self.antenna_pairs}\n")
            f.write("\nMeasurements:\n")
            for meas in measurements:
                f.write(f"  TX{meas['tx_antenna']}→RX{meas['rx_antenna']}: "
                       f"{meas['mean_magnitude_db']:.2f} dB "
                       f"(range: {meas['dynamic_range_db']:.1f} dB)\n")
        
        print(f"📁 Data saved to: {summary_file}")
        print(f"📝 Log saved to: {log_file}")
    
    def cleanup(self):
        """Clean up hardware connections."""
        print("\nCleaning up scanner...")
        self.switch_controller.cleanup()
        self.vna.disconnect()
        print("Scanner cleanup complete")
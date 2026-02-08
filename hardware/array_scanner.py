from switch_controller import RFSwitchController
import serial
import numpy as np

class ArrayScanner:
    def __init__(self, vna_port='COM3'):
        self.switch = RFSwitchController(17, 27, 22, 23)  # GPIO pins
        self.vna = self._connect_vna(vna_port)
        self.antenna_pairs = [(1,3), (1,4), (2,3), (2,4)]  # All 4 combos
    
    def scan_full_array(self, phantom_name="phantom_1"):
        """Scan all antenna pairs and save structured data."""
        all_data = []
        
        for tx, rx in self.antenna_pairs:
            print(f"\n[SCAN] TX Antenna {tx} → RX Antenna {rx}")
            
            # 1. Set switches
            self.switch.select_antenna_pair(tx, rx)
            
            # 2. Collect S21 data from VNA (use your existing VNA code)
            s21_data = self._vna_scan_s21()
            
            # 3. Calculate features in real-time
            features = self._extract_features(s21_data)
            features.update({
                'phantom': phantom_name,
                'tx_antenna': tx,
                'rx_antenna': rx,
                'timestamp': time.time()
            })
            
            all_data.append(features)
            
            print(f"  Average S21: {features['mean_db']:.2f} dB")
            print(f"  Signal Std: {features['std_db']:.2f} dB")
        
        self._save_dataset(all_data, f"{phantom_name}_full_array.csv")
        return all_data
    
    def _extract_features(self, s21_data):
        """Extract ML features from S21 curve."""
        magnitudes = s21_data['magnitude_db']
        return {
            'mean_db': np.mean(magnitudes),
            'std_db': np.std(magnitudes),
            'min_db': np.min(magnitudes),
            'max_db': np.max(magnitudes),
            'range_db': np.max(magnitudes) - np.min(magnitudes),
            'slope': self._calculate_slope(magnitudes),
            'center_freq': 2.5,  # GHz
            'bandwidth': 1.0     # GHz
        }

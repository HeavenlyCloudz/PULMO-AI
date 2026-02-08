import numpy as np
import json
import time
from datetime import datetime
from hardware.switch_controller import RFSwitchController
from hardware.vna_interface import VNAInterface

class ArrayCalibrator:
    """
    Calibrates the complete 4-antenna time-multiplexed array.
    Performs TWO crucial calibrations:
    1. PORT CALIBRATION: OSLT calibration for each antenna through switches
    2. ARRAY CALIBRATION: Baseline measurements for image reconstruction
    """
    
    def __init__(self, scanner):
        """
        Args:
            scanner: ArrayScanner instance with initialized hardware
        """
        self.scanner = scanner
        self.switch = scanner.switch_controller
        self.vna = scanner.vna
        self.calibration_data = {}
        self.calibration_file = None
    
    def full_calibration_procedure(self, cal_name="array_calibration"):
        """
        Complete calibration workflow.
        Follow this EXACT sequence before any imaging scans.
        """
        print("\n" + "="*70)
        print("ARRAY CALIBRATION PROCEDURE")
        print("="*70)
        
        self.calibration_file = f"data/calibration/{cal_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Step 1: Port Calibration (OSLT for each antenna)
        print("\n1. PORT CALIBRATION: OSLT for each antenna through switches")
        print("   You need: Open, Short, Load standards")
        print("   Connect to EACH antenna port when prompted")
        input("   Press Enter to begin port calibration...")
        
        port_cal_data = self._calibrate_all_ports()
        self.calibration_data['port_calibrations'] = port_cal_data
        
        # Step 2: Array Baseline Calibration
        print("\n2. ARRAY BASELINE CALIBRATION")
        print("   Ensure NO objects in scanning area")
        print("   Antennas should be in final positions")
        input("   Press Enter to measure array baseline...")
        
        array_cal_data = self._calibrate_array_baseline()
        self.calibration_data['array_baseline'] = array_cal_data
        
        # Step 3: Through Calibration (Optional but recommended)
        print("\n3. THROUGH CALIBRATION (for transmission normalization)")
        print("   Connect two antennas directly with cable")
        input("   Press Enter for through calibration...")
        
        through_cal_data = self._calibrate_through_paths()
        self.calibration_data['through_calibrations'] = through_cal_data
        
        # Save calibration
        self._save_calibration()
        
        # Verify calibration
        self._verify_calibration()
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print(f"   Calibration saved to: {self.calibration_file}.json")
        print("="*70)
    
    def _calibrate_all_ports(self):
        """
        Perform OSLT calibration for each antenna port THROUGH the switches.
        This is CRITICAL - calibrates at the antenna connectors.
        """
        print("\n" + "-"*50)
        print("PORT CALIBRATION")
        print("-"*50)
        
        port_data = {}
        antennas = [1, 2, 3, 4]  # All 4 antennas
        
        for antenna in antennas:
            print(f"\nCalibrating Antenna {antenna}...")
            
            # Set switches to select this antenna
            if antenna in [1, 2]:
                # TX switch antennas
                tx_port = 1 if antenna == 1 else 2
                self.switch._set_single_switch(self.switch.tx_pins, tx_port)
                # Set RX switch to any position (not used for S11)
                self.switch._set_single_switch(self.switch.rx_pins, 1)
                vna_port = 'CH0'  # TX port for reflection
            else:
                # RX switch antennas
                rx_port = 1 if antenna == 3 else 2
                self.switch._set_single_switch(self.switch.rx_pins, rx_port)
                # Set TX switch to any position
                self.switch._set_single_switch(self.switch.tx_pins, 1)
                vna_port = 'CH1'  # RX port for reflection
            
            input(f"  Connect OPEN standard to Antenna {antenna}, press Enter...")
            open_data = self._measure_s11()
            
            input(f"  Connect SHORT standard to Antenna {antenna}, press Enter...")
            short_data = self._measure_s11()
            
            input(f"  Connect LOAD (50Ω) standard to Antenna {antenna}, press Enter...")
            load_data = self._measure_s11()
            
            # Store calibration data
            port_data[f'antenna_{antenna}'] = {
                'vna_port': vna_port,
                'open': open_data,
                'short': short_data,
                'load': load_data,
                'calibration_time': datetime.now().isoformat()
            }
            
            print(f"  ✓ Antenna {antenna} calibrated")
        
        return port_data
    
    def _calibrate_array_baseline(self):
        """
        Measure baseline S21 for ALL antenna pairs in free space.
        This becomes the reference for detecting anomalies.
        """
        print("\n" + "-"*50)
        print("ARRAY BASELINE MEASUREMENT")
        print("-"*50)
        
        baseline_data = {}
        antenna_pairs = self.switch.get_antenna_pairs()
        
        for tx, rx in antenna_pairs:
            print(f"  Measuring baseline: TX{tx} → RX{rx}")
            
            # Select antenna pair
            self.switch.select_antenna_pair(tx, rx)
            time.sleep(0.05)
            
            # Measure S21
            freqs, mags, phases = self.vna.measure_s21()
            
            baseline_data[f'TX{tx}_RX{rx}'] = {
                'frequencies_hz': freqs.tolist(),
                'baseline_magnitudes_db': mags.tolist(),
                'baseline_phases_deg': phases.tolist(),
                'mean_baseline_db': float(np.mean(mags)),
                'std_baseline_db': float(np.std(mags))
            }
            
            print(f"    Mean S21: {np.mean(mags):.2f} dB, Std: {np.std(mags):.2f} dB")
        
        return baseline_data
    
    def _calibrate_through_paths(self):
        """
        Calibrate direct through connections for transmission normalization.
        Helps normalize cable/switch losses.
        """
        print("\n" + "-"*50)
        print("THROUGH PATH CALIBRATION")
        print("-"*50)
        
        through_data = {}
        
        # You'll need to physically connect antenna pairs with cables
        connections = [
            (1, 3, "Connect Antenna 1 to Antenna 3 with cable"),
            (2, 4, "Connect Antenna 2 to Antenna 4 with cable")
        ]
        
        for tx, rx, instruction in connections:
            print(f"\n{instruction}")
            input("  Press Enter when connected...")
            
            self.switch.select_antenna_pair(tx, rx)
            time.sleep(0.05)
            
            freqs, mags, phases = self.vna.measure_s21()
            
            through_data[f'through_TX{tx}_RX{rx}'] = {
                'frequencies_hz': freqs.tolist(),
                'magnitudes_db': mags.tolist(),
                'phases_deg': phases.tolist(),
                'mean_loss_db': float(np.mean(mags)),
                'cable_loss_db': float(np.mean(mags))  # This is your reference loss
            }
            
            print(f"  ✓ Through path {tx}→{rx}: Mean loss = {np.mean(mags):.2f} dB")
        
        return through_data
    
    def _measure_s11(self):
        """Measure S11 reflection (simplified - in practice use VNA's calibration)."""
        # Note: For accurate S11, you should use the VNA's built-in calibration
        # This is a simplified version for demonstration
        self.vna.ser.write(b'data 0\n')
        time.sleep(0.5)
        
        raw_data = self.vna.ser.read_all().decode('ascii', errors='ignore').strip()
        lines = raw_data.split('\n')
        
        magnitudes = []
        for line in lines:
            if line and not line.startswith('ch>'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        real = float(parts[0])
                        imag = float(parts[1])
                        magnitude = np.sqrt(real**2 + imag**2)
                        magnitude_db = 20 * np.log10(magnitude) if magnitude > 0 else -120
                        magnitudes.append(magnitude_db)
                    except ValueError:
                        continue
        
        return {
            'mean_db': float(np.mean(magnitudes)),
            'std_db': float(np.std(magnitudes)),
            'min_db': float(np.min(magnitudes)),
            'max_db': float(np.max(magnitudes))
        }
    
    def _save_calibration(self):
        """Save calibration data to JSON file."""
        import os
        os.makedirs('data/calibration', exist_ok=True)
        
        # Add metadata
        self.calibration_data['metadata'] = {
            'calibration_name': self.calibration_file,
            'timestamp': datetime.now().isoformat(),
            'frequency_range_ghz': [self.scanner.start_freq/1e9, self.scanner.stop_freq/1e9],
            'points': self.scanner.points,
            'antenna_configuration': '4-antenna time-multiplexed array',
            'rf_switches': 'Mini-Circuits ZFSWA-2-46',
            'vna': 'NanoVNA-F V2'
        }
        
        # Save to JSON
        with open(f'{self.calibration_file}.json', 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        # Also save a simplified version for quick loading
        simplified = {
            'array_baseline_means': {},
            'through_losses': {},
            'antenna_match': {}
        }
        
        # Extract key calibration values
        for key, data in self.calibration_data.get('array_baseline', {}).items():
            simplified['array_baseline_means'][key] = data['mean_baseline_db']
        
        for key, data in self.calibration_data.get('through_calibrations', {}).items():
            simplified['through_losses'][key] = data['mean_loss_db']
        
        for ant in [1, 2, 3, 4]:
            ant_key = f'antenna_{ant}'
            if ant_key in self.calibration_data.get('port_calibrations', {}):
                load_data = self.calibration_data['port_calibrations'][ant_key]['load']
                simplified['antenna_match'][f'ant_{ant}'] = load_data['mean_db']
        
        with open(f'{self.calibration_file}_simple.json', 'w') as f:
            json.dump(simplified, f, indent=2)
    
    def _verify_calibration(self):
        """Verify calibration quality with simple tests."""
        print("\n" + "-"*50)
        print("CALIBRATION VERIFICATION")
        print("-"*50)
        
        # Test 1: Check antenna match (should be <-20 dB for good 50Ω match)
        print("\n1. Antenna Match Check (with 50Ω load):")
        for ant in [1, 2, 3, 4]:
            ant_key = f'antenna_{ant}'
            if ant_key in self.calibration_data.get('port_calibrations', {}):
                load_db = self.calibration_data['port_calibrations'][ant_key]['load']['mean_db']
                status = "✓ GOOD" if load_db < -20 else "⚠️  POOR"
                print(f"   Antenna {ant}: {load_db:.1f} dB {status}")
        
        # Test 2: Check baseline consistency
        print("\n2. Baseline Consistency Check:")
        baseline_means = []
        for key, data in self.calibration_data.get('array_baseline', {}).items():
            baseline_means.append(data['mean_baseline_db'])
            print(f"   {key}: {data['mean_baseline_db']:.1f} dB")
        
        if baseline_means:
            range_db = max(baseline_means) - min(baseline_means)
            if range_db < 10:
                print(f"   ✓ Consistent (range: {range_db:.1f} dB)")
            else:
                print(f"   ⚠️  High variation (range: {range_db:.1f} dB)")
        
        # Test 3: Through path check
        print("\n3. Through Path Loss Check:")
        for key, data in self.calibration_data.get('through_calibrations', {}).items():
            loss = data['mean_loss_db']
            status = "✓ NORMAL" if -5 < loss < 0 else "⚠️  CHECK"
            print(f"   {key}: {loss:.1f} dB {status}")
    
    def load_calibration(self, calibration_file):
        """Load existing calibration data."""
        try:
            with open(calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"✅ Loaded calibration: {calibration_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False
    
    def apply_calibration_to_measurement(self, raw_measurement):
        """
        Apply calibration corrections to a raw measurement.
        
        Args:
            raw_measurement: Dictionary with 'tx_antenna', 'rx_antenna', 's21_magnitudes_db'
            
        Returns: Calibrated measurement
        """
        tx = raw_measurement['tx_antenna']
        rx = raw_measurement['rx_antenna']
        raw_mags = np.array(raw_measurement['s21_magnitudes_db'])
        
        # Get baseline for this antenna pair
        baseline_key = f'TX{tx}_RX{rx}'
        if 'array_baseline' in self.calibration_data and baseline_key in self.calibration_data['array_baseline']:
            baseline = np.array(self.calibration_data['array_baseline'][baseline_key]['baseline_magnitudes_db'])
            
            # Simple calibration: subtract baseline
            calibrated_mags = raw_mags - baseline
            
            # Create calibrated measurement
            calibrated = raw_measurement.copy()
            calibrated['s21_magnitudes_db'] = calibrated_mags.tolist()
            calibrated['mean_magnitude_db'] = float(np.mean(calibrated_mags))
            calibrated['is_calibrated'] = True
            calibrated['calibration_applied'] = 'baseline_subtraction'
            
            return calibrated
        else:
            print(f"⚠️  No calibration data for {baseline_key}, using raw data")
            raw_measurement['is_calibrated'] = False
            return raw_measurement


# ============================================================================
# QUICK CALIBRATION SCRIPT - Run this first!
# ============================================================================

def quick_calibration():
    """Quick calibration script for immediate use."""
    print("\n" + "="*70)
    print("QUICK CALIBRATION SETUP")
    print("="*70)
    print("This will calibrate your array for basic operation.")
    print("For full accuracy, run the complete calibration later.\n")
    
    from scanning.array_scanner import ArrayScanner
    
    # Initialize scanner
    scanner = ArrayScanner()
    
    try:
        scanner.initialize()
        calibrator = ArrayCalibrator(scanner)
        
        print("Choose calibration type:")
        print("1. Quick baseline only (fastest)")
        print("2. Full calibration (recommended, takes 10-15 minutes)")
        
        choice = input("\nSelect (1 or 2): ").strip()
        
        if choice == '1':
            print("\nRunning quick baseline calibration...")
            print("Make sure array area is clear of objects.\n")
            
            # Just measure baselines
            baseline_data = calibrator._calibrate_array_baseline()
            calibrator.calibration_data = {'array_baseline': baseline_data}
            calibrator._save_calibration()
            
            print("\n✅ Quick baseline calibration complete!")
            print("You can now run scans, but for best results,")
            print("run full calibration when you have time.")
            
        elif choice == '2':
            # Run full calibration
            cal_name = input("Enter calibration name (e.g., 'main_cal'): ").strip()
            calibrator.full_calibration_procedure(cal_name)
        
        else:
            print("Invalid choice. Exiting.")
    
    except Exception as e:
        print(f"Calibration failed: {e}")
    
    finally:
        scanner.cleanup()


if __name__ == "__main__":
    # Run this script directly for calibration
    quick_calibration()
import serial
import time
import numpy as np

class VNAInterface:
    """Handles communication with NanoVNA-F V2."""
    
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.frequencies = None
        
    def connect(self):
        """Establish connection to VNA."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Allow connection to settle
            self.ser.reset_input_buffer()
            
            # Test connection
            self.ser.write(b'info\n')
            time.sleep(0.5)
            response = self.ser.read_all().decode('ascii', errors='ignore')
            
            if 'NanoVNA' in response:
                print(f"✅ Connected to VNA on {self.port}")
                print(f"  Device: {response.strip()}")
                return True
            else:
                print("⚠️  Connected but got unexpected response")
                return False
                
        except Exception as e:
            print(f"❌ Failed to connect to VNA: {e}")
            return False
    
    def set_frequency_range(self, start_hz, stop_hz, points=201):
        """Set VNA frequency range."""
        cmd = f"sweep start {start_hz}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.1)
        
        cmd = f"sweep stop {stop_hz}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.1)
        
        cmd = f"sweep points {points}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.1)
        
        # Store frequencies for reference
        self.frequencies = np.linspace(start_hz, stop_hz, points)
        print(f"Frequency set: {start_hz/1e9:.2f} to {stop_hz/1e9:.2f} GHz ({points} points)")
    
    def measure_s21(self):
        """
        Measure S21 transmission.
        Returns: frequencies (Hz), magnitudes (dB), phases (degrees)
        """
        if not self.ser:
            raise ConnectionError("VNA not connected")
        
        # Request S21 data
        self.ser.write(b'data 1\n')
        time.sleep(0.5)
        
        # Read response
        raw_data = self.ser.read_all().decode('ascii', errors='ignore').strip()
        lines = raw_data.split('\n')
        
        magnitudes = []
        phases = []
        
        for line in lines:
            if line and not line.startswith('ch>'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        real = float(parts[0])
                        imag = float(parts[1])
                        
                        # Calculate magnitude in dB
                        magnitude = np.sqrt(real**2 + imag**2)
                        magnitude_db = 20 * np.log10(magnitude) if magnitude > 0 else -120
                        
                        # Calculate phase in degrees
                        phase_rad = np.arctan2(imag, real)
                        phase_deg = np.degrees(phase_rad)
                        
                        magnitudes.append(magnitude_db)
                        phases.append(phase_deg)
                    except ValueError:
                        continue
        
        if not magnitudes:
            raise ValueError("No valid S21 data received")
        
        return self.frequencies, np.array(magnitudes), np.array(phases)
    
    def quick_measure(self):
        """Quick measurement for testing."""
        freqs, mags, phases = self.measure_s21()
        return {
            'frequencies': freqs,
            'magnitudes': mags,
            'mean_magnitude': np.mean(mags),
            'std_magnitude': np.std(mags),
            'min_magnitude': np.min(mags),
            'max_magnitude': np.max(mags)
        }
    
    def disconnect(self):
        """Close VNA connection."""
        if self.ser:
            self.ser.close()
            print("VNA disconnected")
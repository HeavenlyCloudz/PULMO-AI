#!/usr/bin/env python3
"""
PULMO AI - Microwave Array Scanner Main Application
Time-multiplexed 4-antenna imaging system
"""

import time
from datetime import datetime
from scanning.array_scanner import ArrayScanner
from processing.feature_extractor import FeatureExtractor
from processing.image_reconstructor import MicrowaveImageReconstructor

def main():
    print("="*70)
    print("          PULMO AI - MICROWAVE IMAGING ARRAY SCANNER")
    print("="*70)
    
    # Initialize scanner
    scanner = ArrayScanner(
        vna_port='/dev/ttyACM0',  # Change to your VNA port
        tx_pins=[17, 27],         # GPIO pins for TX switch relays
        rx_pins=[22, 23],         # GPIO pins for RX switch relays
        start_freq=2.0e9,         # 2.0 GHz
        stop_freq=3.0e9,          # 3.0 GHz
        points=201                # Frequency points
    )
    
    try:
        # Initialize hardware
        scanner.initialize()
        
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Run full array scan")
            print("2. Scan with current phantom")
            print("3. Scan air (baseline)")
            print("4. Quick test scan")
            print("5. Generate ML dataset")
            print("6. Reconstruct image")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                # Full array scan
                scan_name = input("Enter scan name (e.g., 'phantom_test'): ").strip()
                if not scan_name:
                    scan_name = "full_scan"
                
                measurements = scanner.scan_full_array(scan_name)
                
                # Ask to reconstruct image
                if input("\nReconstruct image? (y/n): ").lower() == 'y':
                    reconstructor = MicrowaveImageReconstructor()
                    fig = reconstructor.plot_reconstruction(
                        measurements, 
                        title=f"Microwave Image: {scan_name}"
                    )
                    plt.savefig(f"data/{scan_name}_image.png", dpi=150)
                    plt.show()
            
            elif choice == '2':
                # Scan with phantom
                phantom_desc = input("Describe phantom (e.g., 'salt_water_2cm_tumor'): ").strip()
                scan_name = f"phantom_{datetime.now().strftime('%H%M%S')}"
                
                print(f"\nPlace {phantom_desc} in center of array")
                input("Press Enter when ready...")
                
                measurements = scanner.scan_full_array(scan_name)
                print(f"\n✅ Phantom scan complete: {phantom_desc}")
            
            elif choice == '3':
                # Air baseline scan
                print("\nRemove all phantoms from array")
                input("Press Enter to scan air baseline...")
                
                measurements = scanner.scan_full_array("air_baseline")
                print("\n✅ Air baseline scan complete")
            
            elif choice == '4':
                # Quick test scan (single pair)
                print("\nQuick test scan (single antenna pair)")
                tx = int(input("TX antenna (1 or 2): "))
                rx = int(input("RX antenna (3 or 4): "))
                
                measurement = scanner.scan_single_pair(tx, rx, "test")
                print(f"\nTest complete: Mean S21 = {measurement['mean_magnitude_db']:.2f} dB")
            
            elif choice == '5':
                # Generate ML dataset
                print("\nGenerating ML dataset from recent scans...")
                # This would load existing scans and create features
                # For now, we'll create from current scan data
                if scanner.scan_data:
                    df = FeatureExtractor.create_ml_dataset(scanner.scan_data)
                    FeatureExtractor.save_feature_dataset(df, "ml_training_dataset.csv")
                else:
                    print("No scan data available. Run a scan first.")
            
            elif choice == '6':
                # Image reconstruction
                if scanner.scan_data:
                    reconstructor = MicrowaveImageReconstructor()
                    fig = reconstructor.plot_reconstruction(
                        scanner.scan_data[-4:],  # Last 4 measurements
                        title="Latest Scan Reconstruction"
                    )
                    plt.show()
                else:
                    print("No scan data available. Run a scan first.")
            
            elif choice == '7':
                print("\nExiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
            
            # Small pause
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        scanner.cleanup()
        print("\n" + "="*70)
        print("Scanner shutdown complete. Goodbye!")
        print("="*70)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick start script for immediate testing.
"""

from scanning.array_scanner import ArrayScanner

def quick_test():
    """Run a quick test scan with minimal setup."""
    print("PULMO AI Quick Test")
    print("Running single antenna pair scan...")
    
    # Create scanner with default settings
    scanner = ArrayScanner()
    
    try:
        scanner.initialize()
        
        # Test single antenna pair
        print("\nTesting antenna pair 1→3...")
        measurement = scanner.scan_single_pair(1, 3, "quick_test")
        
        print(f"\n✅ Quick test complete!")
        print(f"Mean S21: {measurement['mean_magnitude_db']:.2f} dB")
        print(f"Range: {measurement['min_magnitude_db']:.1f} to {measurement['max_magnitude_db']:.1f} dB")
        
        if measurement['mean_magnitude_db'] < -10:
            print("✓ Signal attenuation detected (good!)")
        else:
            print("⚠️  High signal level - check antenna positioning")
    
    finally:
        scanner.cleanup()

if __name__ == "__main__":
    quick_test()
import schedule
import time

def daily_phantom_test():
    """Automated experiment to run every day."""
    scanner = ArrayScanner()
    
    experiments = [
        ("healthy_baseline", "no_phantom"),
        ("phantom_A", "salt_water_only"),
        ("phantom_B", "with_2cm_air_cavity"),
        ("phantom_C", "with_metal_target"),
    ]
    
    for exp_name, phantom_desc in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name} ({phantom_desc})")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*50)
        
        # In real use: prompt user to place phantom
        input(f"Place {phantom_desc} and press Enter...")
        
        data = scanner.scan_full_array(exp_name)
        print(f"✓ Collected {len(data)} antenna pair measurements")
    
    print("\n✅ All experiments completed!")
    print("Data saved to timestamped CSV files.")

# Schedule daily automated scan at 2 PM
schedule.every().day.at("14:00").do(daily_phantom_test)

print("Experiment scheduler started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)

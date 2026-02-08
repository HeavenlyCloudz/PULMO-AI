import numpy as np
import pandas as pd
from scipy import stats
import json

class FeatureExtractor:
    """Extracts ML features from S21 measurements."""
    
    @staticmethod
    def extract_features_from_measurement(measurement):
        """Extract features from a single measurement."""
        mags = measurement['s21_magnitudes_db']
        phases = measurement['s21_phases_deg']
        
        features = {
            # Basic statistics
            'mean_db': np.mean(mags),
            'std_db': np.std(mags),
            'min_db': np.min(mags),
            'max_db': np.max(mags),
            'range_db': np.max(mags) - np.min(mags),
            'median_db': np.median(mags),
            
            # Phase statistics
            'mean_phase': np.mean(phases),
            'phase_std': np.std(phases),
            'phase_range': np.max(phases) - np.min(phases),
            
            # Distribution shape
            'skewness': stats.skew(mags),
            'kurtosis': stats.kurtosis(mags),
            
            # Quartiles
            'q1_db': np.percentile(mags, 25),
            'q3_db': np.percentile(mags, 75),
            'iqr_db': np.percentile(mags, 75) - np.percentile(mags, 25),
            
            # Frequency-dependent features
            'slope': FeatureExtractor._calculate_slope(mags),
            'curvature': FeatureExtractor._calculate_curvature(mags),
            
            # Signal quality
            'signal_to_noise': np.mean(mags) / np.std(mags) if np.std(mags) > 0 else 0,
            'flatness': 1.0 / (1.0 + np.std(mags)),  # Higher = flatter
        }
        
        # Add metadata
        features.update({
            'tx_antenna': measurement['tx_antenna'],
            'rx_antenna': measurement['rx_antenna'],
            'scan_label': measurement.get('scan_label', ''),
            'timestamp': measurement['timestamp']
        })
        
        return features
    
    @staticmethod
    def _calculate_slope(data):
        """Calculate linear slope of the data."""
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    @staticmethod
    def _calculate_curvature(data):
        """Calculate curvature (second derivative estimate)."""
        if len(data) < 3:
            return 0
        # Simple curvature estimate
        first_deriv = np.diff(data)
        second_deriv = np.diff(first_deriv)
        return np.mean(np.abs(second_deriv)) if len(second_deriv) > 0 else 0
    
    @staticmethod
    def create_ml_dataset(measurements, labels=None):
        """
        Create ML-ready dataset from multiple measurements.
        
        Args:
            measurements: List of measurement dictionaries
            labels: Optional list of labels for each measurement
            
        Returns: pandas DataFrame with features
        """
        all_features = []
        
        for i, meas in enumerate(measurements):
            features = FeatureExtractor.extract_features_from_measurement(meas)
            
            # Add label if provided
            if labels is not None and i < len(labels):
                features['target_label'] = labels[i]
            
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        return df
    
    @staticmethod
    def save_feature_dataset(df, filename='ml_dataset.csv'):
        """Save feature dataset to CSV."""
        df.to_csv(filename, index=False)
        print(f"✅ Feature dataset saved to {filename}")
        print(f"   Shape: {df.shape}, Features: {len(df.columns)}")
        
        # Print feature summary
        print("\nFeature summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # First 5 features
            print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
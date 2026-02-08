import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class MicrowaveImageReconstructor:
    """
    Basic microwave image reconstruction from 4-antenna measurements.
    This creates a simple 2D image showing anomaly locations.
    """
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        
        # Define antenna positions (assume square array, 15cm side)
        self.antenna_positions = {
            1: np.array([-7.5, 7.5]),   # Top-left
            2: np.array([7.5, 7.5]),    # Top-right
            3: np.array([-7.5, -7.5]),  # Bottom-left
            4: np.array([7.5, -7.5])    # Bottom-right
        }
        
    def reconstruct_image(self, measurements):
        """
        Reconstruct simple 2D image from measurements.
        
        Args:
            measurements: List of measurement dictionaries
            
        Returns: 2D image array
        """
        # Create grid for reconstruction
        x = np.linspace(-10, 10, self.grid_size)
        y = np.linspace(-10, 10, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize image
        image = np.zeros((self.grid_size, self.grid_size))
        
        # For each measurement, create a simple backprojection
        for meas in measurements:
            tx = meas['tx_antenna']
            rx = meas['rx_antenna']
            magnitude = meas['mean_magnitude_db']
            
            # Get antenna positions
            tx_pos = self.antenna_positions[tx]
            rx_pos = self.antenna_positions[rx]
            
            # Calculate path through each pixel
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Pixel position
                    pixel_pos = np.array([X[i, j], Y[i, j]])
                    
                    # Calculate distances
                    dist_tx = np.linalg.norm(pixel_pos - tx_pos)
                    dist_rx = np.linalg.norm(pixel_pos - rx_pos)
                    total_dist = dist_tx + dist_rx
                    
                    # Simple delay-and-sum reconstruction
                    # Weight contribution based on distance
                    weight = 1.0 / (1.0 + total_dist)  # Closer pixels get higher weight
                    contribution = magnitude * weight
                    
                    # Normalize and add to image
                    image[i, j] += contribution
        
        # Normalize image
        if np.max(np.abs(image)) > 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        return X, Y, image
    
    def plot_reconstruction(self, measurements, title="Microwave Image Reconstruction"):
        """Plot reconstructed image."""
        X, Y, image = self.reconstruct_image(measurements)
        
        plt.figure(figsize=(10, 8))
        
        # Plot image
        plt.imshow(image, extent=[-10, 10, -10, 10], 
                  cmap='viridis', origin='lower', 
                  vmin=0, vmax=1)
        plt.colorbar(label='Normalized Signal Strength')
        
        # Plot antenna positions
        for ant_num, pos in self.antenna_positions.items():
            plt.plot(pos[0], pos[1], 'ro', markersize=10)
            plt.text(pos[0]+0.5, pos[1]+0.5, f'Ant {ant_num}', 
                    color='white', fontweight='bold')
        
        # Plot measurement paths
        for meas in measurements:
            tx = meas['tx_antenna']
            rx = meas['rx_antenna']
            tx_pos = self.antenna_positions[tx]
            rx_pos = self.antenna_positions[rx]
            
            # Draw line between antennas
            plt.plot([tx_pos[0], rx_pos[0]], [tx_pos[1], rx_pos[1]], 
                    'w--', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add measurement info
        info_text = f"Antenna pairs: {len(measurements)}\n"
        info_text += f"Mean S21: {np.mean([m['mean_magnitude_db'] for m in measurements]):.2f} dB"
        plt.text(-9.5, 9, info_text, color='white', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        return plt.gcf()
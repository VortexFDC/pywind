import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram_comparison_lines(df, cols, labels, colors, site='Site', output_dir='output', bins=20, alpha=0.7, save_fig=True):
        """
        Create a histogram comparison plot with lines instead of bars.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the columns to be plotted.
        cols : list
            List of column names to plot.
        labels : list
            List of labels for the legend.
        colors : list
            List of colors for each line.
        site : str
            Site name for the plot title.
        output_dir : str
            Directory to save the output plot.
        bins : int
            Number of bins for the histogram.
        alpha : float
            Transparency of the lines.
        save_fig : bool
            Whether to save the figure or not.
            
        Returns:
        --------
        dict
            Statistics of each distribution.
        """
        plt.figure(figsize=(10, 6))
        
        stats = {}
        
        for col, label, color in zip(cols, labels, colors):
            # Calculate histogram values
            hist_values, bin_edges = np.histogram(df[col].dropna(), bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot as a line
            plt.plot(bin_centers, hist_values, label=label, color=color, linewidth=2, alpha=alpha)
            
            # Store statistics
            stats[label] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
            }
        
        plt.title(f'Wind Speed Distribution Comparison - {site}')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_fig and output_dir and site:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'histogram_comparison_{site}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Histogram line plot saved to {output_dir}/histogram_comparison_{site}.png")
        return stats

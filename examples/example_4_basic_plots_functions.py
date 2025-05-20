import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 
import os
# =============================================================================
# 7. Define Plotting Functions
# =============================================================================

def plot_xy_comparison(df, x_col, y_col, x_label=None, y_label=None, site=None, 
                       output_dir=None, save_fig=True, outlyer_threshold=999, show=True):  
    """
    Create a scatter plot with linear regression comparing two variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data to plot
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data
    x_label : str, optional
        Custom x-axis label (defaults to x_col)
    y_label : str, optional
        Custom y-axis label (defaults to y_col)
    site : str, optional
        Site name for title and filename
    output_dir : str, optional
        Directory to save plot
    save_fig : bool, optional
        Whether to save the figure
    show: bool, optional
        Whether to show the plot
    
    Returns:
    --------
    dict
        Dictionary containing regression statistics
    """
    # Create figure and axis
    plt.figure(figsize=(8, 8))
    
    # Use default labels if not provided
    x_label = x_label or x_col
    y_label = y_label or y_col
    site_name = site.capitalize() if site else ''
    
    # Scatter plot
    plt.scatter(df[x_col], df[y_col], alpha=0.5, color='blue')
    
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
    r_squared = r_value**2
    
    # Create regression line for plotting
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_line = slope * x_line + intercept
    
    # Calculate the expected y values based on regression for each actual x point
    y_expected = slope * df[x_col] + intercept
    y_real= df[y_col]
    df['diff'] = np.abs(y_real - y_expected)

    # Find points with significant differences
    
    outlyers = df['diff'] > outlyer_threshold
    
    # Plot outliers
    plt.scatter(df[x_col][outlyers], df[y_col][outlyers], color='red', alpha=0.5, label='Outliers')
    
    # Plot regression line
    plt.plot(x_line, y_line, 'r-', label=f'y = {slope:.3f}x + {intercept:.3f}')
    
    # Add identity line (perfect agreement)
    plt.plot([0, max(df[x_col].max(), df[y_col].max())], 
             [0, max(df[x_col].max(), df[y_col].max())], 
             'k--', alpha=0.3, label='1:1')
    
    # Add annotations with regression statistics
    plt.annotate(f'$R^2$ = {r_squared:.3f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'Comparison of {x_label} vs {y_label} at {site_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Equal aspect ratio
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig and output_dir and site:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{site}_comparison_{x_col}_vs_{y_col}.png'), dpi=300)
    
    # Show the plot
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()
    
    # Return regression statistics
    stats_dict = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err
    }
    
    return stats_dict


def plot_histogram_comparison(df, cols, labels=None, colors=None, site=None, 
                              output_dir=None, save_fig=True, bins=25, alpha=0.6):
    """
    Create histograms comparing multiple columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data to plot
    cols : list of str
        List of column names to plot
    labels : list of str, optional
        Custom labels for each column (defaults to column names)
    colors : list of str, optional
        Colors for each histogram (defaults to default color cycle)
    site : str, optional
        Site name for title and filename
    output_dir : str, optional
        Directory to save plot
    save_fig : bool, optional
        Whether to save the figure
    bins : int or array-like, optional
        Number of bins or bin edges
    alpha : float, optional
        Transparency for histograms (0-1)
    
    Returns:
    --------
    dict
        Dictionary containing basic statistics for each column
    """
    # Create figure

    plt.figure(figsize=(10, 6))
    
    # Use default labels if not provided
    labels = labels or cols
    colors = colors or ['blue', 'red', 'green', 'orange', 'purple']
    site_name = site.capitalize() if site else ''
    
    # Calculate overall range for binning if bins is an integer
    if isinstance(bins, int):
        max_val = max([df[col].max() for col in cols]) + 1
        bin_edges = np.linspace(0, max_val, bins)
    else:
        bin_edges = bins
    
    # Plot histograms with transparency
    stats_dict = {}
    for i, col in enumerate(cols):
        plt.hist(df[col], bins=bin_edges, alpha=alpha, 
                 label=labels[i], color=colors[i % len(colors)], 
                 edgecolor='black')
        
        
        # Calculate statistics
        mean_val = df[col].mean()
        stats_dict[col] = {
            "mean": mean_val,
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
        
        # Add vertical line for mean
        plt.axvline(mean_val, color=colors[i % len(colors)], 
                    linestyle='dashed', linewidth=1.5)
        
        # Add annotation for mean
        plt.annotate(f'{labels[i]} Mean: {mean_val:.2f}', 
                     xy=(mean_val, plt.ylim()[1] * (0.9 - i*0.1)),
                     xytext=(mean_val + 0.5, plt.ylim()[1] * (0.9 - i*0.1)),
                     arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)]),
                     color=colors[i % len(colors)])
    
    # Add labels and title
    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution Comparison at {site_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig and output_dir and site:
        os.makedirs(output_dir, exist_ok=True)
        col_names = '_'.join([col.replace('_', '') for col in cols])
        plt.savefig(os.path.join(output_dir, f'{site}_histogram_{col_names}.png'), dpi=300)
    
    # Show the plot
    plt.show()
    
    return stats_dict


def plot_annual_means(df, cols, labels=None, colors=None, site=None, 
                      output_dir=None, save_fig=True):
    """
    Create a plot of annual means for given columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index and columns to plot
    cols : list of str
        List of column names to plot
    labels : list of str, optional
        Custom labels for each column (defaults to column names)
    colors : list of str, optional
        Colors for each line (defaults to default color cycle)
    site : str, optional
        Site name for title and filename
    output_dir : str, optional
        Directory to save plot
    save_fig : bool, optional
        Whether to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing annual statistics for each column
    """
    # Ensure DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Use default labels if not provided
    labels = labels or cols
    colors = colors or ['blue', 'red', 'green', 'orange', 'purple']
    site_name = site.capitalize() if site else ''
    
    # Group by month for annual cycle
    monthly_means = {}
    stats_dict = {}
    
    for i, col in enumerate(cols):
        # Group by month and calculate mean
        monthly_data = df[col].groupby(df.index.month).mean()
        monthly_means[col] = monthly_data
        
        # Store stats
        stats_dict[col] = {
            "annual_mean": df[col].mean(),
            "monthly_means": monthly_data.to_dict(),
            "max_month": monthly_data.idxmax(),
            "min_month": monthly_data.idxmin(),
            "annual_std": df[col].std()
        }
        
        # Plot the monthly means
        plt.plot(monthly_data.index, monthly_data.values, 
                 marker='o', linestyle='-', linewidth=2,
                 color=colors[i % len(colors)], label=labels[i])
    
    # Set x-ticks to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), month_names)
    
    # Add labels and title
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'Annual Cycle at {site_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig and output_dir and site:
        os.makedirs(output_dir, exist_ok=True)
        col_names = '_'.join([col.replace('_', '') for col in cols])
        plt.savefig(os.path.join(output_dir, f'{site}_annual_cycle_{col_names}.png'), dpi=300)
    
    # Show the plot
    plt.show()
    
    return stats_dict


def plot_daily_cycle(df, cols, labels=None, colors=None, site=None, 
                     output_dir=None, save_fig=True):
    """
    Create a plot of daily cycle (hour of day) for given columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index and columns to plot
    cols : list of str
        List of column names to plot
    labels : list of str, optional
        Custom labels for each column (defaults to column names)
    colors : list of str, optional
        Colors for each line (defaults to default color cycle)
    site : str, optional
        Site name for title and filename
    output_dir : str, optional
        Directory to save plot
    save_fig : bool, optional
        Whether to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing daily cycle statistics for each column
    """
    # Ensure DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Use default labels if not provided
    labels = labels or cols
    colors = colors or ['blue', 'red', 'green', 'orange', 'purple']
    site_name = site.capitalize() if site else ''
    
    # Group by hour for daily cycle
    hourly_means = {}
    stats_dict = {}
    
    for i, col in enumerate(cols):
        # Group by hour and calculate mean
        hourly_data = df[col].groupby(df.index.hour).mean()
        hourly_means[col] = hourly_data
        
        # Store stats
        stats_dict[col] = {
            "daily_mean": df[col].mean(),
            "hourly_means": hourly_data.to_dict(),
            "max_hour": hourly_data.idxmax(),
            "min_hour": hourly_data.idxmin(),
            "daily_std": hourly_data.std()
        }
        
        # Plot the hourly means
        plt.plot(hourly_data.index, hourly_data.values, 
                 marker='o', linestyle='-', linewidth=2,
                 color=colors[i % len(colors)], label=labels[i])
    
    # Set x-ticks to hour format
    plt.xticks(range(0, 24, 2), [f'{h:02d}:00' for h in range(0, 24, 2)])
    
    # Add labels and title
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'Daily Cycle at {site_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig and output_dir and site:
        os.makedirs(output_dir, exist_ok=True)
        col_names = '_'.join([col.replace('_', '') for col in cols])
        plt.savefig(os.path.join(output_dir, f'{site}_daily_cycle_{col_names}.png'), dpi=300)
    
    # Show the plot
    plt.show()
    
    return stats_dict


def plot_yearly_means(df, cols, labels=None, colors=None, site=None, 
                      output_dir=None, save_fig=True):
    """
    Create a plot of annual means for each year for given columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index and columns to plot
    cols : list of str
        List of column names to plot
    labels : list of str, optional
        Custom labels for each column (defaults to column names)
    colors : list of str, optional
        Colors for each line (defaults to default color cycle)
    site : str, optional
        Site name for title and filename
    output_dir : str, optional
        Directory to save plot
    save_fig : bool, optional
        Whether to save the figure
    
    Returns:
    --------
    dict
        Dictionary containing yearly statistics for each column
    """
    # Ensure DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Use default labels if not provided
    labels = labels or cols
    colors = colors or ['blue', 'red', 'green', 'orange', 'purple']
    site_name = site.capitalize() if site else ''
    
    # Get years in data
    years = df.index.year.unique()
    years.sort_values()
    
    # Group by year for yearly cycle
    yearly_means = {}
    stats_dict = {}
    
    for i, col in enumerate(cols):
        # Group by year and calculate mean
        yearly_data = df[col].groupby(df.index.year).mean()
        yearly_means[col] = yearly_data
        
        # Store stats
        stats_dict[col] = {
            "overall_mean": df[col].mean(),
            "yearly_means": yearly_data.to_dict(),
            "max_year": yearly_data.idxmax(),
            "min_year": yearly_data.idxmin(),
            "yearly_std": yearly_data.std()
        }
        
        # Plot the yearly means
        plt.plot(yearly_data.index, yearly_data.values, 
                 marker='o', linestyle='-', linewidth=2,
                 color=colors[i % len(colors)], label=labels[i])
    
    # Add horizontal lines for overall means
    for i, col in enumerate(cols):
        plt.axhline(y=stats_dict[col]["overall_mean"], 
                   color=colors[i % len(colors)], 
                   linestyle='--', 
                   alpha=0.5,
                   label=f'{labels[i]} Overall Mean')
    
    # Set x-ticks to years
    plt.xticks(years, [str(year) for year in years])
    
    # Add labels and title
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'Yearly Means at {site_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig and output_dir and site:
        os.makedirs(output_dir, exist_ok=True)
        col_names = '_'.join([col.replace('_', '') for col in cols])
        plt.savefig(os.path.join(output_dir, f'{site}_yearly_means_{col_names}.png'), dpi=300)
    
    # Show the plot
    plt.show()
    
    return stats_dict
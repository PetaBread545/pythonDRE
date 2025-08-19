#!/usr/bin/env python3
"""
Example script demonstrating on-demand radius plotting functionality.
This script shows how to create specific radius distribution plots after running the main analysis.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotting import DRGPlotter, load_drg_results


def load_results_from_directory(results_dir: str):
    """
    Load results from a results directory (legacy function for backward compatibility).
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary containing results
    """
    results = {}
    
    # Load complexity scores and convert to expected format
    complexity_scores_path = os.path.join(results_dir, 'complexity_scores.csv')
    if os.path.exists(complexity_scores_path):
        complexity_df = pd.read_csv(complexity_scores_path)
        # Convert to expected format: complexity_table
        results['complexity_table'] = complexity_df.rename(columns={
            'Complexity': 'Complexity',
            'File': 'File', 
            'Likelihood': 'Likelihood'
        })
    
    # Load radius scores and convert to expected format
    radius_scores_path = os.path.join(results_dir, 'radius_scores.csv')
    if os.path.exists(radius_scores_path):
        radius_df = pd.read_csv(radius_scores_path)
        # Convert to expected format: radius_table
        results['radius_table'] = radius_df.rename(columns={
            'Radius': 'Radius',
            'File': 'File',
            'Likelihood': 'Likelihood'
        })
    
    # Load DRG scores and convert to expected format
    drg_scores_path = os.path.join(results_dir, 'drg_scores.csv')
    if os.path.exists(drg_scores_path):
        drg_df = pd.read_csv(drg_scores_path)
        # Convert to expected format: mega_table
        results['mega_table'] = drg_df.rename(columns={
            'File': 'File',
            'Likelihood_of_De_Identifying': 'Likelihood_of_De_Identifying'
        })
    
    # Create empty distance_tables since they're not currently saved
    # This is a limitation - distance tables would need to be saved during the main analysis
    results['distance_tables'] = {}
    
    # Load distance tables if available
    distance_tables_path = os.path.join(results_dir, 'distance_tables.pkl')
    if os.path.exists(distance_tables_path):
        import pickle
        with open(distance_tables_path, 'rb') as f:
            results['distance_tables'] = pickle.load(f)
        print(f"  Loaded {len(results['distance_tables'])} distance tables")
    else:
        print("  No distance tables found (distance_tables.pkl not present)")
    
    print("Loaded results structure:")
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key}: {value.shape} DataFrame")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")
    
    return results


def main():
    """Main function to demonstrate on-demand radius plotting."""
    
    # Configuration
    results_file = "results_test_01/drg_results.pkl"  # Path to the saved results file
    plots_dir = "plots_test_01"      # Change this to your plots directory
    
    print("DRG On-Demand Radius Plotting Example")
    print("=" * 50)
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Results file '{results_file}' not found.")
        print("Please run the main analysis first using: python main.py --config config.json")
        return
    
    # Load results using the new function
    print(f"Loading results from {results_file}...")
    try:
        results = load_drg_results(results_file)
        print("✓ Results loaded successfully")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Create plotter
    plotter = DRGPlotter(results, plots_dir)
    
    # Get available plotting options
    print("\nAvailable plotting options:")
    options = plotter.get_available_plotting_options()
    print(options)
    print(f"Available files: {options['available_files']}")
    print(f"Available complexities: {options['available_complexities']}")
    print(f"Available radii: {options['available_radii']}")
    print(f"Number of distance tables: {len(options['distance_tables_keys'])}")
    
    if options['distance_tables_keys']:
        print("\nSample distance table keys:")
        for i, key in enumerate(options['distance_tables_keys'][:5]):  # Show first 5
            print(f"  {i+1}. {key}")
        if len(options['distance_tables_keys']) > 5:
            print(f"  ... and {len(options['distance_tables_keys']) - 5} more")
    
    # Check if distance tables are available for radius distribution plots
    if not options['distance_tables_keys']:
        print("\n" + "=" * 50)
        print("WARNING: Distance tables not available for radius distribution plots")
        print("=" * 50)
        print("The radius distribution plots require distance tables that are not currently saved.")
        print("Creating summary plots instead...")
        
        # Create summary plots instead
        print("\nCreating summary score plots...")
        try:
            fig = plotter.plot_summary_scores(save=True)
            print(f"Summary plots saved to {plots_dir}")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating summary plots: {e}")
        
        # Try to create radius bar plot
        try:
            fig = plotter._plot_radius_bar(save=True)
            print(f"Radius bar plot saved to {plots_dir}")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating radius bar plot: {e}")
        
        print("\nNote: Radius distribution plots require distance tables to be saved during the main analysis.")
        print("The main analysis now automatically saves distance tables for on-demand plotting.")
        
        return
    
    # Example 1: Plot radius distribution for specific parameters
    print("\n" + "=" * 50)
    print("Example 1: Plot radius distribution for specific parameters")
    print("=" * 50)
    
    # Get the first available file
    if options['available_files']:
        filename = options['available_files'][0]
        print(f"Creating radius plot for file: {filename}")
        
        # Create plot with default parameters (first available)
        fig = plotter.plot_on_demand_radius(
            filename=filename,
            save=True
        )
        print(f"Plot saved to {plots_dir}")
        plt.close(fig)
    
    # Example 2: Plot radius distribution for specific complexity
    print("\n" + "=" * 50)
    print("Example 2: Plot radius distribution for specific complexity")
    print("=" * 50)
    
    if options['available_complexities']:
        complexity = options['available_complexities'][0]
        print(f"Creating radius plot for complexity: {complexity}")
        
        fig = plotter.plot_on_demand_radius(
            complexity=complexity,
            save=True
        )
        print(f"Plot saved to {plots_dir}")
        plt.close(fig)
    
    # Example 3: Plot radius distribution for specific radius
    print("\n" + "=" * 50)
    print("Example 3: Plot radius distribution for specific radius")
    print("=" * 50)
    
    if options['available_radii']:
        radius = options['available_radii'][0]
        print(f"Creating radius plot for radius: {radius}")
        
        fig = plotter.plot_on_demand_radius(
            radius=radius,
            save=True
        )
        print(f"Plot saved to {plots_dir}")
        plt.close(fig)
    
    # Example 4: Plot radius distribution with all parameters specified
    print("\n" + "=" * 50)
    print("Example 4: Plot radius distribution with all parameters specified")
    print("=" * 50)
    
    if (options['available_files'] and options['available_complexities'] and 
        options['available_radii']):
        filename = options['available_files'][0]
        complexity = options['available_complexities'][0]
        radius = options['available_radii'][0]
        
        print(f"Creating radius plot for:")
        print(f"  File: {filename}")
        print(f"  Complexity: {complexity}")
        print(f"  Radius: {radius}")
        
        fig = plotter.plot_on_demand_radius(
            filename=filename,
            complexity=complexity,
            radius=radius,
            save=True
        )
        print(f"Plot saved to {plots_dir}")
        plt.close(fig)
    
    # Example 5: Interactive plotting
    print("\n" + "=" * 50)
    print("Example 5: Interactive plotting")
    print("=" * 50)
    print("You can also create plots interactively:")
    print("""
# Example usage:
plotter = DRGPlotter(results, "plots_dir")

# Plot for specific file
fig = plotter.plot_on_demand_radius(filename="demo_new_copulas.csv")

# Plot for specific complexity
fig = plotter.plot_on_demand_radius(complexity=3)

# Plot for specific radius
fig = plotter.plot_on_demand_radius(radius=0.1)

# Plot for specific target ID
fig = plotter.plot_on_demand_radius(target_id=100)

# Plot with multiple parameters
fig = plotter.plot_on_demand_radius(
    filename="demo_new_copulas.csv",
    complexity=5,
    radius=0.3,
    target_id=50
)

# Get available options
options = plotter.get_available_plotting_options()
print(options)
    """)
    
    print("\nOn-demand radius plotting completed!")
    print(f"Check the '{plots_dir}' directory for generated plots.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()

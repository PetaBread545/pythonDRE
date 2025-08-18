"""
Plotting module for DRG analysis results.
Generates the same plots as the R Shiny application.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List
import os
from pathlib import Path
import json
import pickle

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DRGPlotter:
    """
    Class for generating DRG analysis plots.
    """
    
    def __init__(self, results: Dict, output_dir: str = "plots"):
        """
        Initialize the plotter with results.
        
        Args:
            results: Dictionary containing DRG analysis results
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Save results metadata for on-demand plotting
        self._save_results_metadata()
        
    def _save_results_metadata(self):
        """Save metadata about available results for on-demand plotting."""
        metadata = {
            'available_files': [],
            'available_complexities': [],
            'available_radii': [],
            'distance_tables_keys': []
        }
        
        # Extract available files
        if 'mega_table' in self.results:
            metadata['available_files'] = self.results['mega_table']['File'].tolist()
        
        # Extract available complexities
        if 'complexity_table' in self.results:
            metadata['available_complexities'] = sorted(self.results['complexity_table']['Complexity'].unique().tolist())
        
        # Extract available radii
        if 'radius_table' in self.results:
            metadata['available_radii'] = sorted(self.results['radius_table']['Radius'].unique().tolist())
        
        # Extract distance table keys
        if 'distance_tables' in self.results and self.results['distance_tables']:
            metadata['distance_tables_keys'] = list(self.results['distance_tables'].keys())
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'plotting_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save distance tables separately for on-demand access
        if 'distance_tables' in self.results and self.results['distance_tables']:
            distance_data_path = os.path.join(self.output_dir, 'distance_tables.pkl')
            with open(distance_data_path, 'wb') as f:
                pickle.dump(self.results['distance_tables'], f)
        
    def plot_scores(self, save: bool = True) -> plt.Figure:
        """
        Create bar plot of DRG scores for each file.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        mega_table = self.results['mega_table']
        
        # Filter out error messages and convert to numeric
        valid_data = mega_table.copy()
        valid_data['Likelihood_of_De_Identifying'] = pd.to_numeric(
            valid_data['Likelihood_of_De_Identifying'], errors='coerce'
        )
        valid_data = valid_data.dropna()
        
        if len(valid_data) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No valid scores to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('DRG Scores by File')
            return fig
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(valid_data)), valid_data['Likelihood_of_De_Identifying'], 
                     color='steelblue', alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('File')
        ax.set_ylabel('Disclosure Risk')
        ax.set_title('DRG Scores by File')
        ax.set_xticks(range(len(valid_data)))
        ax.set_xticklabels(valid_data['File'], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/drg_scores.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_complexity(self, save: bool = True) -> plt.Figure:
        """
        Create bar plot of DRG scores by complexity.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        complexity_table = self.results['complexity_table']
        
        if len(complexity_table) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No complexity data to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('DRG Scores by Complexity')
            return fig
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by complexity and file
        pivot_data = complexity_table.pivot(index='File', columns='Complexity', values='Likelihood')
        
        # Create grouped bar plot
        x = np.arange(len(pivot_data))
        width = 0.25
        
        for i, complexity in enumerate(pivot_data.columns):
            bars = ax.bar(x + i * width, pivot_data[complexity], width, 
                         label=f'Complexity {complexity}', alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('File')
        ax.set_ylabel('Disclosure Risk')
        ax.set_title('DRG Scores by Complexity')
        ax.set_xticks(x + width * (len(pivot_data.columns) - 1) / 2)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/complexity_scores.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_radius(self, save: bool = True) -> plt.Figure:
        """
        Create histogram with KDE plot of distance distribution for radius analysis.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get distance data from the analyzer's distance tables
        if not hasattr(self, 'results') or 'distance_tables' not in self.results:
            # Fallback to original bar plot if no distance data available
            return self._plot_radius_bar(save)
        
        # Get the first available distance table for visualization
        distance_tables = self.results.get('distance_tables', {})
        if not distance_tables:
            return self._plot_radius_bar(save)
        
        # Use the first distance table available
        first_key = list(distance_tables.keys())[0]
        distance_data = distance_tables[first_key]
        
        # Extract radius and target distance from the key
        # Key format: "filename_x_target_p_set_radius"
        key_parts = first_key.split('_')
        try:
            radius = float(key_parts[-1])
            target_distance = distance_data[distance_data['Casej'] == distance_data['Casei'].iloc[0]]['Distance_Perc'].iloc[0] if len(distance_data) > 0 else 0.0
        except (ValueError, IndexError):
            radius = 0.5  # Default radius
            target_distance = 0.0  # Default target distance
        
        # Get selected features for the title
        features_selected = self.results.get('features_selected', [])
        if features_selected:
            feature_names = " | ".join([str(f) for f in features_selected[0]])
        else:
            feature_names = "Default features"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create histogram with density
        ax.hist(distance_data['Distance_Perc'], bins=50, density=True, 
               alpha=0.7, color='steelblue', label='Distance Distribution')
        
        # Add KDE curve
        kde = stats.gaussian_kde(distance_data['Distance_Perc'])
        x_range = np.linspace(distance_data['Distance_Perc'].min(), 
                             distance_data['Distance_Perc'].max(), 100)
        ax.plot(x_range, kde(x_range), 'gray', linewidth=2, label='KDE')
        
        # Add vertical lines for radius and target distance
        ax.axvline(x=radius, color='blue', linestyle='--', linewidth=2, 
                  label=f'Radius ({radius})')
        ax.axvline(x=target_distance, color='red', linestyle='--', linewidth=2, 
                  label=f'Target_Distance ({target_distance:.3f})')
        
        # Customize the plot
        ax.set_xlabel('Distance_Perc')
        ax.set_ylabel('density')
        ax.set_title(f'Features Selected: {feature_names}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to match the image (0 to 1)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/radius_distribution.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_on_demand_radius(self, filename: str = None, complexity: int = None, 
                             radius: float = None, target_id: int = None, 
                             save: bool = True) -> plt.Figure:
        """
        Create radius distribution plot on demand with specific parameters.
        
        Args:
            filename: Name of the obfuscated file to plot
            complexity: Complexity (number of features) to plot
            radius: Radius parameter to plot
            target_id: Target ID to plot
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Load distance tables if not already loaded
        distance_tables = self.results.get('distance_tables', {})
        if not distance_tables:
            # Try to load from saved file
            distance_data_path = os.path.join(self.output_dir, 'distance_tables.pkl')
            if os.path.exists(distance_data_path):
                with open(distance_data_path, 'rb') as f:
                    distance_tables = pickle.load(f)
        
        # Check if distance tables are available
        if not distance_tables:
            # Create informative error plot
            fig, ax = plt.subplots(figsize=(12, 8))
            error_msg = (
                "Distance tables not available for radius distribution plots.\n\n"
                "This functionality requires distance tables to be saved during the main analysis.\n"
                "Currently, only summary scores are available.\n\n"
                "Available data:\n"
                f"- Files: {len(self.results.get('mega_table', pd.DataFrame()).get('File', []))}\n"
                f"- Complexities: {len(self.results.get('complexity_table', pd.DataFrame()).get('Complexity', []))}\n"
                f"- Radii: {len(self.results.get('radius_table', pd.DataFrame()).get('Radius', []))}\n\n"
                "Use plot_scores() for summary plots instead."
            )
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_title('Distance Tables Not Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            if save:
                plot_filename = "radius_distribution_error_no_distance_tables.png"
                plt.savefig(f"{self.output_dir}/{plot_filename}", dpi=300, bbox_inches='tight')
            
            return fig
        
        # Find matching distance table
        matching_key = None
        for key in distance_tables.keys():
            # Parse key components (format: filename_target_complexity_radius)
            # Handle filenames with underscores by finding the last 3 parts
            key_parts = key.split('_')
            
            if len(key_parts) >= 4:
                # Extract the last 3 parts: target, complexity, radius
                target_part = key_parts[-3]
                complexity_part = key_parts[-2]
                radius_part = key_parts[-1]
                
                # Reconstruct filename (everything before the last 3 parts)
                filename_part = '_'.join(key_parts[:-3])
                
                # Parse values
                key_target = int(target_part) if target_part.isdigit() else None
                key_complexity = int(complexity_part) if complexity_part.isdigit() else None
                key_radius = float(radius_part) if radius_part.replace('.', '').isdigit() else None
                
                # Check if this key matches our criteria
                if (filename is None or filename_part == filename) and \
                   (target_id is None or key_target == target_id) and \
                   (complexity is None or key_complexity == complexity) and \
                   (radius is None or abs(key_radius - radius) < 1e-6):
                    matching_key = key
                    break
        
        if matching_key is None:
            # Create a plot showing available options
            fig, ax = plt.subplots(figsize=(12, 8))
            available_keys = list(distance_tables.keys())[:10]  # Show first 10 keys
            error_msg = (
                f'No matching data found for:\n'
                f'Filename: {filename}\n'
                f'Complexity: {complexity}\n'
                f'Radius: {radius}\n'
                f'Target ID: {target_id}\n\n'
                f'Available distance table keys ({len(distance_tables)} total):\n'
                + '\n'.join([f'  - {key}' for key in available_keys])
                + (f'\n  ... and {len(distance_tables) - 10} more' if len(distance_tables) > 10 else '')
            )
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            ax.set_title('No Matching Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            if save:
                plot_filename = "radius_distribution_error_no_match.png"
                plt.savefig(f"{self.output_dir}/{plot_filename}", dpi=300, bbox_inches='tight')
            
            return fig
        
        # Get the distance data
        distance_data = distance_tables[matching_key]
        
        # Extract parameters from the key using the same logic as above
        key_parts = matching_key.split('_')
        target_part = key_parts[-3]
        complexity_part = key_parts[-2]
        radius_part = key_parts[-1]
        filename_part = '_'.join(key_parts[:-3])
        
        actual_radius = float(radius_part) if len(key_parts) > 3 else 0.5
        actual_target = int(target_part) if len(key_parts) > 1 else 0
        actual_complexity = int(complexity_part) if len(key_parts) > 2 else 0
        
        # Find target distance
        target_distance = 0.0
        if len(distance_data) > 0:
            target_row = distance_data[distance_data['Casej'] == distance_data['Casei'].iloc[0]]
            if len(target_row) > 0:
                target_distance = target_row['Distance_Perc'].iloc[0]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create histogram with density
        ax.hist(distance_data['Distance_Perc'], bins=50, density=True, 
               alpha=0.7, color='steelblue', label='Distance Distribution')
        
        # Add KDE curve
        if len(distance_data) > 1:
            try:
                kde = stats.gaussian_kde(distance_data['Distance_Perc'])
                x_range = np.linspace(distance_data['Distance_Perc'].min(), 
                                     distance_data['Distance_Perc'].max(), 100)
                ax.plot(x_range, kde(x_range), 'gray', linewidth=2, label='KDE')
            except:
                pass  # Skip KDE if there are issues
        
        # Add vertical lines for radius and target distance
        ax.axvline(x=actual_radius, color='blue', linestyle='--', linewidth=2, 
                  label=f'Radius ({actual_radius})')
        ax.axvline(x=target_distance, color='red', linestyle='--', linewidth=2, 
                  label=f'Target Distance ({target_distance:.3f})')
        
        # Customize the plot
        ax.set_xlabel('Normalized Distance')
        ax.set_ylabel('Density')
        ax.set_title(f'Distance Distribution\nFile: {key_parts[0]}, Target: {actual_target}, Complexity: {actual_complexity}, Radius: {actual_radius}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            # Create filename based on parameters
            safe_filename = key_parts[0].replace('/', '_').replace('\\', '_')
            plot_filename = f"radius_distribution_{safe_filename}_target{actual_target}_comp{actual_complexity}_rad{actual_radius}.png"
            plt.savefig(f"{self.output_dir}/{plot_filename}", dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_available_plotting_options(self) -> Dict:
        """
        Get available options for on-demand plotting.
        
        Returns:
            Dictionary with available files, complexities, radii, and targets
        """
        metadata_path = os.path.join(self.output_dir, 'plotting_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'available_files': [],
                'available_complexities': [],
                'available_radii': [],
                'distance_tables_keys': []
            }
    
    def _plot_radius_bar(self, save: bool = True) -> plt.Figure:
        """
        Create bar plot of DRG scores by radius (fallback method).
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        radius_table = self.results['radius_table']
        
        if len(radius_table) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No radius data to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('DRG Scores by Radius')
            return fig
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by radius and file
        pivot_data = radius_table.pivot(index='File', columns='Radius', values='Likelihood')
        
        # Create grouped bar plot
        x = np.arange(len(pivot_data))
        width = 0.25
        
        for i, radius in enumerate(pivot_data.columns):
            bars = ax.bar(x + i * width, pivot_data[radius], width, 
                         label=f'Radius {radius}', alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('File')
        ax.set_ylabel('Disclosure Risk')
        ax.set_title('DRG Scores by Radius')
        ax.set_xticks(x + width * (len(pivot_data.columns) - 1) / 2)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/radius_scores.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distance_distribution(self, distance_data: pd.DataFrame, target_distance: float, 
                                 radius: float, save: bool = True) -> plt.Figure:
        """
        Create histogram and density plot of distances with radius and target distance markers.
        
        Args:
            distance_data: DataFrame containing distance data
            target_distance: Distance of the target
            radius: Radius parameter
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram with density
        ax.hist(distance_data['Distance_Perc'], bins=50, density=True, 
               alpha=0.7, color='steelblue', label='Distance Distribution')
        
        # Add density curve
        kde = stats.gaussian_kde(distance_data['Distance_Perc'])
        x_range = np.linspace(distance_data['Distance_Perc'].min(), 
                             distance_data['Distance_Perc'].max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density')
        
        # Add vertical lines for radius and target distance
        ax.axvline(x=radius, color='blue', linestyle='--', linewidth=2, 
                  label=f'Radius ({radius})')
        ax.axvline(x=target_distance, color='red', linestyle='--', linewidth=2, 
                  label=f'Target Distance ({target_distance:.3f})')
        
        # Customize the plot
        ax.set_xlabel('Normalized Distance')
        ax.set_ylabel('Density')
        ax.set_title('Distance Distribution with Radius and Target Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/distance_distribution.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_all_plots(self) -> Dict[str, plt.Figure]:
        """
        Create all standard plots for the DRG analysis (excluding radius distributions).
        
        Returns:
            Dictionary mapping plot names to figures
        """
        plots = {}
        
        # Create main plots (excluding radius distribution)
        plots['scores'] = self.plot_scores()
        plots['complexity'] = self.plot_complexity()
        plots['radius_bar'] = self._plot_radius_bar()
        
        return plots
    
    def save_all_plots(self):
        """Save all standard plots to the output directory (excluding radius distributions)."""
        self.create_all_plots()
        plt.close('all')  # Close all figures to free memory
        print(f"All standard plots saved to {self.output_dir}")
        print("Use plot_on_demand_radius() for specific radius distribution plots")

    def plot_summary_scores(self, save: bool = True) -> plt.Figure:
        """
        Create summary plots of DRG scores using available data.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DRG Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall DRG scores by file
        if 'mega_table' in self.results and not self.results['mega_table'].empty:
            ax1 = axes[0, 0]
            mega_table = self.results['mega_table']
            files = mega_table['File']
            scores = pd.to_numeric(mega_table['Likelihood_of_De_Identifying'], errors='coerce')
            
            bars = ax1.bar(range(len(files)), scores, color='steelblue', alpha=0.8)
            ax1.set_xlabel('File')
            ax1.set_ylabel('Disclosure Risk')
            ax1.set_title('DRG Scores by File')
            ax1.set_xticks(range(len(files)))
            ax1.set_xticklabels(files, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if not pd.isna(score):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                            f'{score:.4f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 0].text(0.5, 0.5, 'No mega table data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('DRG Scores by File')
        
        # Plot 2: Complexity scores
        if 'complexity_table' in self.results and not self.results['complexity_table'].empty:
            ax2 = axes[0, 1]
            complexity_table = self.results['complexity_table']
            
            # Group by complexity and calculate mean
            complexity_means = complexity_table.groupby('Complexity')['Likelihood'].mean()
            
            bars = ax2.bar(complexity_means.index, complexity_means.values, color='lightcoral', alpha=0.8)
            ax2.set_xlabel('Complexity (Number of Features)')
            ax2.set_ylabel('Average Disclosure Risk')
            ax2.set_title('DRG Scores by Complexity')
            
            # Add value labels
            for bar, score in zip(bars, complexity_means.values):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'No complexity table data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('DRG Scores by Complexity')
        
        # Plot 3: Radius scores
        if 'radius_table' in self.results and not self.results['radius_table'].empty:
            ax3 = axes[1, 0]
            radius_table = self.results['radius_table']
            
            # Group by radius and calculate mean
            radius_means = radius_table.groupby('Radius')['Likelihood'].mean()
            
            bars = ax3.bar(radius_means.index, radius_means.values, color='lightgreen', alpha=0.8)
            ax3.set_xlabel('Radius')
            ax3.set_ylabel('Average Disclosure Risk')
            ax3.set_title('DRG Scores by Radius')
            
            # Add value labels
            for bar, score in zip(bars, radius_means.values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No radius table data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('DRG Scores by Radius')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary_text = "Summary Statistics:\n\n"
        
        if 'mega_table' in self.results and not self.results['mega_table'].empty:
            scores = pd.to_numeric(self.results['mega_table']['Likelihood_of_De_Identifying'], errors='coerce')
            scores = scores.dropna()
            if len(scores) > 0:
                summary_text += f"Overall Scores:\n"
                summary_text += f"  Mean: {scores.mean():.4f}\n"
                summary_text += f"  Std: {scores.std():.4f}\n"
                summary_text += f"  Min: {scores.min():.4f}\n"
                summary_text += f"  Max: {scores.max():.4f}\n"
                summary_text += f"  Count: {len(scores)}\n\n"
        
        if 'complexity_table' in self.results and not self.results['complexity_table'].empty:
            summary_text += f"Complexities: {sorted(self.results['complexity_table']['Complexity'].unique())}\n"
        
        if 'radius_table' in self.results and not self.results['radius_table'].empty:
            summary_text += f"Radii: {sorted(self.results['radius_table']['Radius'].unique())}\n"
        
        if 'distance_tables' in self.results:
            summary_text += f"Distance Tables: {len(self.results['distance_tables'])}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save:
            plot_filename = "drg_summary_plots.png"
            plt.savefig(f"{self.output_dir}/{plot_filename}", dpi=300, bbox_inches='tight')
        
        return fig


def create_summary_report(results: Dict, output_dir: str = "reports") -> str:
    """
    Create a summary report of the DRG analysis.
    
    Args:
        results: DRG analysis results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f"{output_dir}/drg_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("DRG Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        mega_table = results['mega_table']
        valid_scores = pd.to_numeric(mega_table['Likelihood_of_De_Identifying'], errors='coerce')
        valid_scores = valid_scores.dropna()
        
        f.write(f"Total files analyzed: {len(mega_table)}\n")
        f.write(f"Valid files: {len(valid_scores)}\n")
        f.write(f"Mean disclosure risk: {valid_scores.mean():.4f}\n")
        f.write(f"Standard deviation: {valid_scores.std():.4f}\n")
        f.write(f"Minimum risk: {valid_scores.min():.4f}\n")
        f.write(f"Maximum risk: {valid_scores.max():.4f}\n\n")
        
        # File-specific results
        f.write("File-specific Results:\n")
        f.write("-" * 30 + "\n")
        for _, row in mega_table.iterrows():
            f.write(f"{row['File']}: {row['Likelihood_of_De_Identifying']}\n")
        
    
    return report_path

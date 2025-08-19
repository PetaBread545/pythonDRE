# Python DRG (Disclosure Risk Generator) Module

A Python implementation of the DRG algorithm for estimating disclosure risk in obfuscated datasets. This module provides the same functionality as the R Shiny application but as a command-line tool with modern Python libraries.

## Features

- **Core DRG Algorithm**: Complete implementation of the disclosure risk generator algorithm
- **Multiple File Support**: Analyze multiple obfuscated datasets simultaneously
- **Configurable Parameters**: JSON-based configuration for all analysis parameters
- **Comprehensive Plotting**: Generate the same plots as the R application
- **Advanced Radius Distribution Plotting**: Create detailed radius distribution plots with distance tables
- **Distance Table Storage**: Save and load distance tables for on-demand analysis
- **Modern Python Stack**: Uses pandas, numpy, scikit-learn, matplotlib, and seaborn
- **Command-line Interface**: Easy-to-use CLI with progress tracking
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Installation

1. **Clone or download the project**:
   ```bash
   cd pythonDRE
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Create sample data** (for testing):
   ```bash
   python main.py --create-sample-data
   ```

2. **Run the analysis**:
   ```bash
   python main.py --config config.json
   ```

3. **View results**:
   - Results are saved in the `results_dir` directory specified in `config.json`
   - Plots are saved in the `plots_dir` directory specified in `config.json`
   - Distance tables are saved for on-demand plotting

## Advanced Radius Distribution Plotting

The module now includes advanced radius distribution plotting capabilities that save distance tables during analysis and allow for detailed post-analysis visualization.

### Distance Table Storage

During the main analysis, the system automatically saves:
- **Distance Tables (Pickle)**: `distance_tables.pkl` - Efficient binary storage for plotting
- **Distance Tables (CSV)**: `distance_tables_csv/` directory - Human-readable format for inspection
- **Plotting Metadata**: `plotting_metadata.json` - Available options for plotting

### Basic Usage

```python
from plotting import DRGPlotter, load_drg_results

# Load results from saved file
results = load_drg_results("results_analysis/drg_results.pkl")

# Create plotter
plotter = DRGPlotter(results, "plots_dir")

# Get available plotting options
options = plotter.get_available_plotting_options()
print(f"Available files: {options['available_files']}")
print(f"Available complexities: {options['available_complexities']}")
print(f"Available radii: {options['available_radii']}")
print(f"Available targets: {len(options['distance_tables_keys'])} distance tables")

# Create specific radius plots
fig = plotter.plot_on_demand_radius(filename="demo_new_copulas.csv")
fig = plotter.plot_on_demand_radius(complexity=3)
fig = plotter.plot_on_demand_radius(radius=0.1)
fig = plotter.plot_on_demand_radius(target_id=100)

# Plot with multiple parameters
fig = plotter.plot_on_demand_radius(
    filename="demo_new_copulas.csv",
    complexity=5,
    radius=0.3,
    target_id=50
)
```

### Example Script

Run the example script to see advanced plotting in action:

```bash
python example_radius_plotting.py
```

This script demonstrates:
- Loading results from the saved `drg_results.pkl` file
- Loading distance tables for detailed plotting
- Getting available plotting options
- Creating various types of radius distribution plots
- Interactive plotting examples
- Error handling when distance tables are not available

### Loading Results from Previous Analyses

You can load and analyze results from previous DRG analyses using the saved results file:

```python
from plotting import DRGPlotter, load_drg_results

# Load results from a previous analysis
results = load_drg_results("results_previous_analysis/drg_results.pkl")

# Create plotter with loaded results
plotter = DRGPlotter(results, "new_plots_directory")

# Generate plots using the loaded data
fig = plotter.plot_summary_scores(save=True)
fig = plotter.plot_on_demand_radius(
    filename="my_data.csv",
    complexity=5,
    radius=0.1,
    save=True
)
```

### Alternative Loading Method (Legacy)

For backward compatibility, you can also load results from individual CSV files:

```python
from example_radius_plotting import load_results_from_directory
from plotting import DRGPlotter

# Load results from individual files
results = load_results_from_directory("results_previous_analysis")

# Create plotter with loaded results
plotter = DRGPlotter(results, "new_plots_directory")
```

### Plot Features

Each radius distribution plot includes:
- **Histogram**: Distribution of distances from target
- **KDE Curve**: Kernel density estimation overlay
- **Radius Marker**: Vertical line showing the radius threshold
- **Target Distance**: Vertical line showing the target's distance
- **Statistics**: Mean, standard deviation, and other metrics
- **Metadata**: File, target ID, complexity, and radius information

### Summary Plots

The system also generates comprehensive summary plots:
- **DRG Scores by File**: Overall disclosure risk scores
- **DRG Scores by Complexity**: Risk scores across different feature counts
- **DRG Scores by Radius**: Risk scores across different radius values
- **Summary Statistics**: Statistical overview of the analysis

### Plot Storage

The plotting system stores:
- **Metadata**: `plotting_metadata.json` - Available files, complexities, radii, and targets
- **Distance Tables**: `distance_tables.pkl` - Distance data for on-demand plotting
- **CSV Files**: `distance_tables_csv/` - Individual distance tables for inspection
- **Plots**: PNG files with descriptive names based on parameters

### Distance Table Format

Distance tables contain:
- **Casej**: Target case identifier
- **Casei**: Reference case identifier  
- **Distance**: Raw Gower distance
- **Distance_Perc**: Normalized distance (0-1)
- **Neighbors_x**: Neighbor identifier

### Error Handling

The plotting system provides informative error messages:
- When distance tables are not available
- When no matching data is found for specified parameters
- When data loading fails
- With suggestions for alternative plotting options

## Configuration

The analysis is configured via a JSON file (`config.json`):

```json
{
  "input_files": {
    "original_file": "data/original_data.csv",
    "obfuscated_files": [
      "data/obfuscated_1.csv",
      "data/obfuscated_2.csv",
      "data/obfuscated_3.csv"
    ]
  },
  "parameters": {
    "replications": 10,
    "target_selection": {
      "method": "random",
      "targets": [20, 50, 100]
    },
    "feature_selection": {
      "method": "random",
      "complexities": [5, 10, 15],
      "features": ["feature1", "feature2", "feature3"]
    },
    "radius": [0.1, 0.3, 0.5]
  },
  "output": {
    "results_dir": "results",
    "plots_dir": "plots",
    "save_plots": true,
    "save_results": true
  }
}
```

### Configuration Parameters

- **`original_file`**: Path to the original dataset (CSV format)
- **`obfuscated_files`**: List of paths to obfuscated datasets
- **`replications`**: Number of replications for the analysis
- **`target_selection`**:
  - `method`: "random" or "specified"
  - `targets`: List of target row indices (used if method is "specified")
- **`feature_selection`**:
  - `method`: "random" or "specified"
  - `complexities`: List of feature counts to test
  - `features`: List of specific feature names (used if method is "specified")
- **`radius`**: List of radius values to test (0-1 range)

## Usage Examples

### Basic Analysis
```bash
python main.py --config config.json
```

### Create Sample Data and Run
```bash
python main.py --create-sample-data
python main.py --config config.json
```

### Skip Plot Generation
```bash
python main.py --config config.json --no-plots
```

### Custom Output Directory
```bash
python main.py --config config.json --output-dir my_results
```

### Verbose Logging
```bash
python main.py --config config.json --verbose
```

## Output Files

### Results Directory
- `drg_results.pkl`: Complete analysis results (recommended for loading)
- `drg_scores.csv`: Main DRG scores for each file
- `complexity_scores.csv`: Scores broken down by complexity
- `radius_scores.csv`: Scores broken down by radius
- `config.json`: Copy of the configuration used
- `drg_summary_report.txt`: Text summary of the analysis
- `distance_tables.pkl`: Distance tables for on-demand plotting
- `distance_tables_csv/`: Directory containing individual distance tables as CSV files
- `plotting_metadata.json`: Available plotting options and metadata

### Plots Directory
- `drg_scores.png/png`: Bar plot of DRG scores by file
- `complexity_scores.png/png`: Bar plot of scores by complexity
- `radius_scores.png/png`: Bar plot of scores by radius
- `drg_summary_plots.png`: Comprehensive summary plots
- `radius_distribution_*.png`: On-demand radius distribution plots
- `plotting_metadata.json`: Available plotting options

## Data Format Requirements

### Input Data
- **Format**: CSV files
- **ID Column**: Optional (will be added automatically if missing)
- **Mixed Data Types**: Supports both numerical and categorical variables
- **Compatibility**: Original and obfuscated files must have the same structure

### Example Data Structure
```csv
ID,feature_1,feature_2,feature_3,category,region
1,0.5,1.2,-0.3,A,North
2,0.8,0.9,0.1,B,South
3,-0.2,1.5,0.7,C,East
...
```

## Algorithm Details

The DRG algorithm works as follows:

1. **Feature Selection**: Randomly or specifically select features for analysis
2. **Distance Calculation**: Compute Gower distances between data points
3. **Neighborhood Definition**: Define neighborhoods based on radius parameter
4. **Risk Assessment**: Calculate likelihood of target identification
5. **Aggregation**: Average results across replications and parameter combinations

## Performance Considerations

- **Large Datasets**: The algorithm scales with O(n²) complexity due to distance calculations
- **Memory Usage**: Distance matrices can be memory-intensive for large datasets
- **Parallelization**: Consider running multiple analyses in parallel for different parameter sets

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure all file paths in config.json are correct
2. **Memory Errors**: Reduce dataset size or number of replications
3. **Shape Mismatch**: Ensure original and obfuscated files have identical structure
4. **Import Errors**: Make sure virtual environment is activated and dependencies are installed

### Debug Mode
```bash
python main.py --config config.json --verbose
```

## API Usage

You can also use the modules programmatically:

```python
from drg_analyzer import DRGAnalyzer
from plotting import DRGPlotter

# Initialize analyzer
analyzer = DRGAnalyzer('config.json')

# Run analysis
results = analyzer.run_analysis()

# Generate plots
plotter = DRGPlotter(results, 'plots/')
plotter.save_all_plots()

# Get summary statistics
summary = analyzer.get_summary_stats()
print(summary)
```

## Acknowledgments

This Python implementation is based on the original R Shiny application for DRG analysis. The algorithm and methodology remain the same, but the implementation uses modern Python libraries for improved performance and maintainability.

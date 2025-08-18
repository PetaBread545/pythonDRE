#!/usr/bin/env python3
"""
Main script for running DRG (Disclosure Risk Generator) analysis.
This script provides a command-line interface to run the complete DRG analysis.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drg_analyzer import DRGAnalyzer
from plotting import DRGPlotter, create_summary_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> bool:
    """
    Validate the configuration file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'input_files',
        'parameters',
        'output'
    ]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Check input files
    if 'original_file' not in config['input_files']:
        logger.error("Missing original_file in input_files")
        return False
    
    if 'obfuscated_files' not in config['input_files']:
        logger.error("Missing obfuscated_files in input_files")
        return False
    
    if not config['input_files']['obfuscated_files']:
        logger.error("No obfuscated files specified")
        return False
    
    # Check parameters
    required_params = ['replications', 'target_selection', 'feature_selection', 'radius']
    for param in required_params:
        if param not in config['parameters']:
            logger.error(f"Missing required parameter: {param}")
            return False
    
    # Check file existence
    original_file = config['input_files']['original_file']
    if not os.path.exists(original_file):
        logger.error(f"Original file not found: {original_file}")
        return False
    
    for obf_file in config['input_files']['obfuscated_files']:
        if not os.path.exists(obf_file):
            logger.error(f"Obfuscated file not found: {obf_file}")
            return False
    
    return True


def create_sample_data():
    """
    Create sample data files for testing.
    """
    import pandas as pd
    import numpy as np
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate sample original data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create original data
    original_data = pd.DataFrame()
    original_data['ID'] = range(1, n_samples + 1)
    
    # Add numerical features
    for i in range(n_features):
        original_data[f'feature_{i+1}'] = np.random.normal(0, 1, n_samples)
    
    # Add categorical features
    original_data['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    original_data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    original_data.to_csv('data/original_data.csv', index=False)
    
    # Create obfuscated versions
    for i in range(3):
        obfuscated_data = original_data.copy()
        
        # Add noise to numerical features
        for j in range(n_features):
            noise = np.random.normal(0, 0.1, n_samples)
            obfuscated_data[f'feature_{j+1}'] += noise
        
        # Sometimes change categorical values
        mask = np.random.random(n_samples) < 0.1
        obfuscated_data.loc[mask, 'category'] = np.random.choice(['A', 'B', 'C'], mask.sum())
        
        obfuscated_data.to_csv(f'data/obfuscated_{i+1}.csv', index=False)
    
    print("Sample data created in 'data/' directory")


def main():
    """Main function to run DRG analysis."""
    parser = argparse.ArgumentParser(
        description='DRG (Disclosure Risk Generator) Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.json
  python main.py --config config.json --create-sample-data
  python main.py --config config.json --no-plots
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )
    
    parser.add_argument(
        '--create-sample-data',
        default=False,
        action='store_true',
        help='Create sample data files for testing'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample data...")
        create_sample_data()
        logger.info("Sample data created successfully")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Use --create-sample-data to create sample data and config")
        return 1
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        if not validate_config(config):
            logger.error("Configuration validation failed")
            return 1
        
        # Override output directory if specified
        if args.output_dir:
            config['output']['results_dir'] = args.output_dir
            config['output']['plots_dir'] = args.output_dir
        
        # Create analyzer and run analysis
        logger.info("Initializing DRG analyzer...")
        analyzer = DRGAnalyzer(args.config)
        
        logger.info("Running DRG analysis...")
        results = analyzer.run_analysis()
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_results()
        
        # Generate plots
        if not args.no_plots:
            logger.info("Generating plots...")
            plotter = DRGPlotter(results, config['output']['plots_dir'])
            plotter.save_all_plots()
        
        # Create summary report
        logger.info("Creating summary report...")
        report_path = create_summary_report(results, config['output']['results_dir'])
        
        # Print summary
        summary = analyzer.get_summary_stats()
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {config['output']['results_dir']}")
        if not args.no_plots:
            logger.info(f"Plots saved to: {config['output']['plots_dir']}")
        logger.info(f"Summary report: {report_path}")
        
        print("\n" + "="*50)
        print("DRG ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total files analyzed: {summary['total_files']}")
        print(f"Valid files: {summary['valid_files']}")
        print(f"Mean disclosure risk: {summary['mean_likelihood']:.4f}")
        print(f"Standard deviation: {summary['std_likelihood']:.4f}")
        print(f"Risk range: {summary['min_likelihood']:.4f} - {summary['max_likelihood']:.4f}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
DRG Analyzer - Main class for orchestrating DRG analysis across multiple files and parameters.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import os

from drg_core import drg, validate_parameters, DRGResult

logger = logging.getLogger(__name__)


class DRGAnalyzer:
    """
    Main analyzer class for DRG (Disclosure Risk Generator) analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the DRG analyzer with configuration.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.results = {}
        self.distance_tables = {}
        self.random_targets = []
        self.features_selected = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def _validate_file_compatibility(self, d0: pd.DataFrame, d1: pd.DataFrame, file_name: str) -> Tuple[bool, str]:
        """Validate that obfuscated file is compatible with original file."""
        if d0.shape != d1.shape:
            return False, f"Shape mismatch in {file_name}: D0 {d0.shape} vs D1 {d1.shape}"
        
        if list(d0.columns) != list(d1.columns):
            return False, f"Column mismatch in {file_name}"
        
        return True, ""
    
    def _get_targets(self, d0: pd.DataFrame, method: str, targets: List[int]) -> List[int]:
        """Get target indices based on method."""
        if method == "random":
            # Generate random targets for each replication
            return []
        else:
            # Use specified targets
            return [t for t in targets if 1 <= t <= len(d0)]
    
    def _get_features(self, d0: pd.DataFrame, method: str, complexities: List[int], 
                     features: List[str]) -> List[int]:
        """Get feature indices based on method."""
        if method == "random":
            return complexities
        else:
            # Convert feature names to indices
            feature_indices = []
            for feature in features:
                if feature in d0.columns:
                    feature_indices.append(d0.columns.get_loc(feature))
            return feature_indices
    
    def run_analysis(self) -> Dict:
        """
        Run the complete DRG analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting DRG analysis...")
        
        # Load original data
        d0 = self._load_data(self.config['input_files']['original_file'])
        
        # Initialize results containers
        n_files = len(self.config['input_files']['obfuscated_files'])
        replications = self.config['parameters']['replications']
        complexities = self.config['parameters']['feature_selection']['complexities']
        radii = self.config['parameters']['radius']
        
        # Initialize results tables
        mega_table = pd.DataFrame({
            'File': [Path(f).name for f in self.config['input_files']['obfuscated_files']],
            'Likelihood_of_De_Identifying': [0.0] * n_files
        })
        
        complexity_table = []
        radius_table = []
        
        # Process each obfuscated file
        for i, obfuscated_file in enumerate(tqdm(self.config['input_files']['obfuscated_files'], 
                                               desc="Processing files")):
            try:
                d1 = self._load_data(obfuscated_file)
                
                # Validate file compatibility
                is_valid, error_msg = self._validate_file_compatibility(d0, d1, obfuscated_file)
                if not is_valid:
                    mega_table.iloc[i, 1] = error_msg
                    continue
                
                # Get analysis parameters
                target_method = self.config['parameters']['target_selection']['method']
                feature_method = self.config['parameters']['feature_selection']['method']
                
                file_score = 0.0
                file_complexity_scores = {comp: 0.0 for comp in complexities}
                file_radius_scores = {radius: 0.0 for radius in radii}
                
                # Run analysis for each combination of parameters
                total_operations = len(complexities) * len(radii) * replications
                operation_count = 0
                
                for p_set in complexities:
                    for delta in radii:
                        for rep in range(replications):
                            # Select target
                            if target_method == "random":
                                x_target = np.random.randint(1, len(d0) + 1)
                                self.random_targets.append(x_target)
                            else:
                                targets = self.config['parameters']['target_selection']['targets']
                                x_target = targets[rep % len(targets)]
                            
                            try:
                                # Run DRG analysis
                                result = drg(
                                    x_target=x_target,
                                    p_set=p_set,
                                    d0=d0,
                                    delta=delta,
                                    d1=d1,
                                    random_feature=(feature_method == "random"),
                                    features=None  # Will be randomly selected if random_feature=True
                                )
                                
                                # Store results
                                likelihood = result.likelihood
                                selected_features = result.selected_features
                                
                                file_score += likelihood
                                file_complexity_scores[p_set] += likelihood
                                file_radius_scores[delta] += likelihood
                                
                                # Store distance table
                                key = f"{Path(obfuscated_file).name}_{x_target}_{p_set}_{delta}"
                                self.distance_tables[key] = result.d1_distances
                                
                                # Store selected features if random
                                if feature_method == "random":
                                    self.features_selected.append(selected_features)
                                
                                operation_count += 1
                                
                            except Exception as e:
                                logger.error(f"Error in DRG calculation: {e}")
                                continue
                
                # Calculate average scores
                if operation_count > 0:
                    mega_table.iloc[i, 1] = file_score / operation_count
                    
                    # Store complexity scores
                    for comp in complexities:
                        complexity_table.append({
                            'Complexity': comp,
                            'File': Path(obfuscated_file).name,
                            'Likelihood': file_complexity_scores[comp] / (len(radii) * replications)
                        })
                    
                    # Store radius scores
                    for radius in radii:
                        radius_table.append({
                            'Radius': radius,
                            'File': Path(obfuscated_file).name,
                            'Likelihood': file_radius_scores[radius] / (len(complexities) * replications)
                        })
                
            except Exception as e:
                logger.error(f"Error processing file {obfuscated_file}: {e}")
                mega_table.iloc[i, 1] = f"Error: {str(e)}"
        
        # Convert to DataFrames
        complexity_df = pd.DataFrame(complexity_table)
        radius_df = pd.DataFrame(radius_table)
        
        # Store results
        self.results = {
            'mega_table': mega_table,
            'complexity_table': complexity_df,
            'radius_table': radius_df,
            'distance_tables': self.distance_tables,
            'features_selected': self.features_selected,
            'config': self.config
        }
        
        logger.info("DRG analysis completed successfully")
        return self.results
    
    def save_results(self, output_dir: str = None):
        """Save analysis results to files."""
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        self.results['mega_table'].to_csv(f"{output_dir}/drg_scores.csv", index=False)
        self.results['complexity_table'].to_csv(f"{output_dir}/complexity_scores.csv", index=False)
        self.results['radius_table'].to_csv(f"{output_dir}/radius_scores.csv", index=False)
        
        # Save complete results in pickle format for easy loading
        import pickle
        results_file_path = f"{output_dir}/drg_results.pkl"
        with open(results_file_path, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Saved complete results to {results_file_path}")
        
        # Save distance tables for plotting functionality
        if self.distance_tables:
            distance_tables_path = f"{output_dir}/distance_tables.pkl"
            with open(distance_tables_path, 'wb') as f:
                pickle.dump(self.distance_tables, f)
            logger.info(f"Saved {len(self.distance_tables)} distance tables to {distance_tables_path}")
            
            # Also save distance tables in CSV format for easier inspection
            distance_tables_dir = f"{output_dir}/distance_tables_csv"
            os.makedirs(distance_tables_dir, exist_ok=True)
            
            for key, distance_df in self.distance_tables.items():
                # Create safe filename
                safe_filename = key.replace('/', '_').replace('\\', '_') + '.csv'
                csv_path = os.path.join(distance_tables_dir, safe_filename)
                distance_df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved {len(self.distance_tables)} distance tables as CSV to {distance_tables_dir}")
        else:
            logger.warning("No distance tables to save")
        
        # Save configuration
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the analysis."""
        if not self.results:
            return {}
        
        mega_table = self.results['mega_table']
        valid_scores = pd.to_numeric(mega_table['Likelihood_of_De_Identifying'], errors='coerce')
        valid_scores = valid_scores.dropna()
        
        return {
            'total_files': len(mega_table),
            'valid_files': len(valid_scores),
            'mean_likelihood': valid_scores.mean() if len(valid_scores) > 0 else 0,
            'std_likelihood': valid_scores.std() if len(valid_scores) > 0 else 0,
            'min_likelihood': valid_scores.min() if len(valid_scores) > 0 else 0,
            'max_likelihood': valid_scores.max() if len(valid_scores) > 0 else 0
        }

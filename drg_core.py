"""
Core DRG (Disclosure Risk Generator) algorithm implementation.
This module contains the main DRG function and supporting utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Union, Optional
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DRGResult:
    """Result object for DRG calculations."""
    likelihood: float = 0.0
    d0_neighbors: List[int] = None
    d1_neighbors: List[int] = None
    selected_features: List[int] = None
    exact_matches: List[int] = None
    d0_distances: pd.DataFrame = None
    d1_distances: pd.DataFrame = None
    
    def __post_init__(self):
        if self.d0_neighbors is None:
            self.d0_neighbors = []
        if self.d1_neighbors is None:
            self.d1_neighbors = []
        if self.selected_features is None:
            self.selected_features = []
        if self.exact_matches is None:
            self.exact_matches = []


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data for DRG analysis.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Preprocessed DataFrame with ID column and encoded categorical variables
    """
    # Make a copy to avoid modifying original data
    data_processed = data.copy()
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = data_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if data_processed[col].isnull().any():
            median_val = data_processed[col].median()
            data_processed[col] = data_processed[col].fillna(median_val)
    
    # For categorical columns, fill with mode
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data_processed[col].isnull().any():
            mode_val = data_processed[col].mode().iloc[0] if len(data_processed[col].mode()) > 0 else "Unknown"
            data_processed[col] = data_processed[col].fillna(mode_val)
    
    # Add ID column if not present or if first column is not sequential
    if 'ID' not in data_processed.columns:
        # Check if first column looks like an ID
        first_col = data_processed.columns[0]
        if first_col.lower().find('id') != -1 or first_col.lower().find('participant') != -1:
            # Rename first column to ID and ensure it starts from 1
            data_processed = data_processed.rename(columns={first_col: 'ID'})
            # Ensure ID starts from 1 and is sequential
            data_processed['ID'] = range(1, len(data_processed) + 1)
        else:
            # Insert ID column at the beginning
            data_processed.insert(0, 'ID', range(1, len(data_processed) + 1))
    else:
        # Ensure ID starts from 1 and is sequential
        data_processed['ID'] = range(1, len(data_processed) + 1)
    
    # Encode categorical variables
    label_encoders = {}
    
    for col in data_processed.columns:
        if col == 'ID':  # Skip ID column
            continue
        if data_processed[col].dtype == 'object':
            # Handle any remaining NaN values in categorical columns
            data_processed[col] = data_processed[col].fillna("Unknown")
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col].astype(str))
            label_encoders[col] = le
    
    return data_processed, label_encoders


def calculate_gower_distance(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate Gower distance matrix that matches R's cluster::daisy() implementation.
    Optimized using numpy.
    
    Args:
        data: DataFrame with mixed data types
        
    Returns:
        Distance matrix matching R's implementation
    """
    # Ensure no NaN values remain
    data_clean = data.fillna(data.median() if data.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
    
    # Separate numerical and categorical columns
    numerical_cols = data_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = data_clean.select_dtypes(include=['object']).columns
    
    n_samples = len(data_clean)
    distances = np.zeros((n_samples, n_samples))
    
    # Calculate numerical distances (Gower's)
    if len(numerical_cols) > 0:
        numerical_data = data_clean[numerical_cols].values.astype(float)
        
        # Normalize each numerical column by its range
        for i in range(numerical_data.shape[1]):
            col_data = numerical_data[:, i]
            col_range = col_data.max() - col_data.min()
            
            if col_range > 0:
                # Normalize by range
                normalized_col = col_data / col_range
                numerical_data[:, i] = normalized_col
            else:
                # Constant column - set to 0
                numerical_data[:, i] = 0
        
        # Calculate Manhattan distance for normalized numerical data
        numerical_distances = pairwise_distances(numerical_data, metric='manhattan')
        distances += numerical_distances
    
    # Calculate categorical distances
    if len(categorical_cols) > 0:
        categorical_data = data_clean[categorical_cols].values
        
        # Calculate Hamming distance for categorical features
        categorical_distances = pairwise_distances(categorical_data, metric='hamming')
        
        # Scale by number of categorical variables
        categorical_distances = categorical_distances * len(categorical_cols)
        distances += categorical_distances
    
    # Normalize by total number of variables
    total_vars = len(numerical_cols) + len(categorical_cols)
    if total_vars > 0:
        distances = distances / total_vars
    
    # Ensure no NaN values in final distance matrix
    distances = np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)
    
    return distances


def find_exact_matches(data: pd.DataFrame, target_row: int, feature_indices: List[int]) -> List[int]:
    """
    Find rows that exactly match the target row for the specified features.
    
    Args:
        data: Input DataFrame
        target_row: Index of target row
        feature_indices: List of feature column indices
        
    Returns:
        List of row indices that match exactly
    """
    # Ensure target_row is within bounds
    if target_row < 0 or target_row >= len(data):
        return []
    
    # Get target values, handling potential NaN values
    target_values = data.iloc[target_row, feature_indices].values
    
    # Convert to string for comparison to handle mixed types
    target_values_str = [str(val) if pd.notna(val) else "NaN" for val in target_values]
    
    matches = []
    
    for i in range(len(data)):
        if i != target_row:
            row_values = data.iloc[i, feature_indices].values
            row_values_str = [str(val) if pd.notna(val) else "NaN" for val in row_values]
            
            # Check for exact match
            if target_values_str == row_values_str:
                matches.append(i)
    
    return matches


def drg(x_target: int, p_set: int, d0: pd.DataFrame, delta: float, d1: pd.DataFrame, 
         random_feature: bool = True, features: List[int] = None) -> DRGResult:
    """
    Calculate Disclosure Risk Generator (DRG) metric.
    
    Args:
        x_target: Target row index (0-based)
        p_set: Number of features to select
        d0: Original dataset
        delta: Radius parameter
        d1: Obfuscated dataset
        random_feature: Whether to select features randomly
        features: List of specific features to use (if not random)
        
    Returns:
        DRGResult object with likelihood and metadata
    """
    logger.debug(f"DRG calculation: target={x_target}, p_set={p_set}, delta={delta}")
    
    # Preprocess data
    d0_processed, _ = preprocess_data(d0.copy())
    d1_processed, _ = preprocess_data(d1.copy())
    
    # Validate parameters
    if x_target >= len(d0_processed):
        raise ValueError(f"Target index {x_target} out of range (max: {len(d0_processed)-1})")
    
    if p_set < 1 or p_set >= d0_processed.shape[1]:
        raise ValueError(f"Invalid p_set: {p_set} (must be 1 to {d0_processed.shape[1]-1})")
    
    if delta < 0 or delta > 1:
        raise ValueError(f"Invalid delta: {delta} (must be 0 to 1)")
    
    # Select features - FIXED: match R implementation exactly
    if random_feature:
        # R: p <- sort(sample(2:dim(D0)[2],p_set))
        # In R, this selects from columns 2 to n (1-based indexing)
        # In Python, this translates to columns 1 to n-1 (0-based indexing)
        available_features = list(range(1, d0_processed.shape[1]))  # Skip ID column (column 0)
        p_set = min(p_set, len(available_features))
        if p_set > 0:
            selected_features = np.random.choice(available_features, size=p_set, replace=False)
            selected_features = sorted(selected_features)  # Sort like R does
        else:
            selected_features = [1]  # Default to first feature if no features available
    else:
        selected_features = features if features else list(range(1, min(p_set + 1, d0_processed.shape[1])))
    
    logger.debug(f"Selected features: {selected_features}")
    
    # Create subset of D0 with selected features - FIXED: match R exactly
    # R: D0_p <- D0[,p]
    d0_p = d0_processed.iloc[:, selected_features]
    
    # Calculate distances for D0 - FIXED: use subset, not full dataset
    # R: D0_distances_x0 <- as.matrix(daisy(D0_p, metric = "gower"))[x_target,]
    d0_distances = calculate_gower_distance(d0_p)
    d0_distances_x = d0_distances[x_target, :]
    
    # Create distance dataframe for D0 - match R structure exactly
    # R: D0_distances_x <- data.frame(Casej = 1:dim(D0_p)[1], Casei = x_target, Distance = D0_distances_x0)[-x_target,]
    d0_distances_df = pd.DataFrame({
        'Casej': range(1, len(d0_p) + 1),
        'Casei': x_target + 1,  # R uses 1-based indexing
        'Distance': d0_distances_x
    })
    d0_distances_df = d0_distances_df[d0_distances_df['Casej'] != (x_target + 1)]
    d0_distances_df = d0_distances_df.sort_values('Distance')
    
    # Normalize distances for D0 - match R exactly
    # R: D0_distances_x_sorted$Distance_Perc <- (D0_distances_x_sorted$Distance - D0_distances_x_sorted$Distance[1]) / (max(D0_distances_x_sorted$Distance) - D0_distances_x_sorted$Distance[1])
    min_dist_d0 = d0_distances_df['Distance'].min()
    max_dist_d0 = d0_distances_df['Distance'].max()
    if max_dist_d0 > min_dist_d0:
        d0_distances_df['Distance_Perc'] = (d0_distances_df['Distance'] - min_dist_d0) / (max_dist_d0 - min_dist_d0)
    else:
        d0_distances_df['Distance_Perc'] = 0
    
    # Add Neighbors_x column - match R exactly
    # R: D0_distances_x_sorted$Neighbors_x <- D0_distances_x_sorted[,1]
    d0_distances_df['Neighbors_x'] = d0_distances_df['Casej']
    
    # Fix Neighbors_x for target - match R exactly
    # R: for (i in 1:length(D0_distances_x_sorted$Neighbors_x)) { if (D0_distances_x_sorted$Neighbors_x[i]==x_target) { D0_distances_x_sorted$Neighbors_x[i] <- D0_distances_x_sorted[i,2] } }
    for i in range(len(d0_distances_df)):
        if d0_distances_df.iloc[i]['Neighbors_x'] == (x_target + 1):
            d0_distances_df.iloc[i, d0_distances_df.columns.get_loc('Neighbors_x')] = d0_distances_df.iloc[i]['Casei']
    
    # Find exact matches in D0 - FIXED: match R exactly
    # R: x_match = NULL; for (i in 1:dim(D0)[1]) { if (length(which((D0[i,p]==D0[x_target,p])==TRUE))==length(p)) { x_match <- c(x_match,i) } }
    exact_matches = []
    target_values = d0_processed.iloc[x_target, selected_features].values
    
    for i in range(len(d0_processed)):
        if i != x_target:  # Don't include target itself
            row_values = d0_processed.iloc[i, selected_features].values
            # Check if all selected features match exactly
            if np.array_equal(target_values, row_values):
                exact_matches.append(i)
    
    # Create D1_new - FIXED: match R exactly
    # R: D1_new <- rbind(D1,D0[x_match,])
    # Note: R only adds exact matches, NOT the target separately
    d1_new = d1_processed.copy()
    if exact_matches:
        exact_matches_data = d0_processed.iloc[exact_matches]
        d1_new = pd.concat([d1_new, exact_matches_data], ignore_index=True)
    
    # Add target row from D0 to D1_new if it's not already there
    # This is needed for the distance calculation
    target_row_d0 = d0_processed.iloc[[x_target]]
    d1_new = pd.concat([d1_new, target_row_d0], ignore_index=True)
    
    # Create subset of D1_new with selected features - match R exactly
    # R: D1_p <- D1_new[,p]
    d1_p_new = d1_new.iloc[:, selected_features]
    
    # Calculate distances for D1 - FIXED: use subset, not full dataset
    # R: D1_distances_x0 = as.matrix(daisy(D1_p, metric = "gower"))[dim(D1_new)[1],]
    d1_distances = calculate_gower_distance(d1_p_new)
    # Target is the last row in D1_new
    target_row_in_d1 = len(d1_new) - 1
    d1_distances_x = d1_distances[target_row_in_d1, :]
    
    # Create distance dataframe for D1 - match R exactly
    # R: D1_distances_x = data.frame(Casej = 1:dim(D1_new)[1], Casei = dim(D1_new)[1],Distance = D1_distances_x0)[-dim(D1_new)[1],]
    d1_distances_df = pd.DataFrame({
        'Casej': range(1, len(d1_p_new) + 1),
        'Casei': len(d1_p_new),
        'Distance': d1_distances_x
    })
    d1_distances_df = d1_distances_df[d1_distances_df['Casej'] != len(d1_p_new)]
    d1_distances_df = d1_distances_df.sort_values('Distance')
    
    # Normalize distances for D1 - match R exactly
    # R: D1_distances_x_sorted$Distance_Perc <- (D1_distances_x_sorted$Distance - D1_distances_x_sorted$Distance[1]) / (max(D1_distances_x_sorted$Distance) - D1_distances_x_sorted$Distance[1])
    min_dist_d1 = d1_distances_df['Distance'].min()
    max_dist_d1 = d1_distances_df['Distance'].max()
    if max_dist_d1 > min_dist_d1:
        d1_distances_df['Distance_Perc'] = (d1_distances_df['Distance'] - min_dist_d1) / (max_dist_d1 - min_dist_d1)
    else:
        d1_distances_df['Distance_Perc'] = 0
    
    # Add Neighbors_x column for D1 - match R exactly
    # R: D1_distances_x_sorted$Neighbors_x <- D1_distances_x_sorted[,1]
    d1_distances_df['Neighbors_x'] = d1_distances_df['Casej']
    
    # Fix Neighbors_x for target in D1 - match R exactly
    # R: for (i in 1:length(D1_distances_x_sorted$Neighbors_x)) { if (D1_distances_x_sorted$Neighbors_x[i]==dim(D1_new)[1]) { D1_distances_x_sorted$Neighbors_x[i] <- D1_distances_x_sorted[i,2] } }
    for i in range(len(d1_distances_df)):
        if d1_distances_df.iloc[i]['Neighbors_x'] == len(d1_new):
            d1_distances_df.iloc[i, d1_distances_df.columns.get_loc('Neighbors_x')] = d1_distances_df.iloc[i]['Casei']
    
    # Calculate neighbors based on delta - FIXED: use Neighbors_x column
    # R: x_neighbors_D0 <- as.numeric(head(D0_distances_x_sorted, n=length(which(DRG$D0_neighbors[,4]<delta)))[,5])
    # R: x_neighbors_D1 <- as.numeric(head(D1_distances_x_sorted, n=length(which(DRG$D1_neighbors[,4]<delta)))[,5])
    d0_neighbors = d0_distances_df[d0_distances_df['Distance_Perc'] < delta]['Neighbors_x'].tolist()
    d1_neighbors = d1_distances_df[d1_distances_df['Distance_Perc'] < delta]['Neighbors_x'].tolist()
    
    # Calculate likelihood metric - FIXED: match R exactly
    # R: a <- which(DRG$D1_neighbors[1:length(which(DRG$D1_neighbors[,4]<delta)),5]==x_target)
    # R: if (identical(a,integer(0))) { DRG$metric1 <- 0 } else { DRG$metric1 <- 1/length(which(DRG$D1_neighbors[,4]<delta)) }
    if len(d1_neighbors) > 0:
        # Check if the target (original target ID) is in the neighbors using Neighbors_x column
        target_in_neighbors = (x_target + 1) in d1_neighbors  # Convert to 1-based indexing
        if target_in_neighbors:
            likelihood = 1.0 / len(d1_neighbors)
            logger.debug(f"Target found in neighbors. Likelihood: {likelihood}")
        else:
            likelihood = 0.0
            logger.debug(f"Target NOT found in neighbors. Likelihood: 0.0")
    else:
        likelihood = 0.0
        logger.debug(f"No neighbors found. Likelihood: 0.0")
    
    # Debug logging
    logger.debug(f"Target: {x_target + 1}, Delta: {delta}")
    logger.debug(f"D0 neighbors: {len(d0_neighbors)}")
    logger.debug(f"D1 neighbors: {len(d1_neighbors)}")
    logger.debug(f"Target in D1 neighbors: {(x_target + 1) in d1_neighbors}")
    logger.debug(f"Exact matches found: {len(exact_matches)}")
    logger.debug(f"D1_new size: {len(d1_new)} (original D1: {len(d1_processed)})")
    logger.debug(f"Selected features: {selected_features}")
    
    return DRGResult(
        likelihood=likelihood,
        d0_neighbors=d0_neighbors,
        d1_neighbors=d1_neighbors,
        selected_features=selected_features,
        exact_matches=exact_matches,
        d0_distances=d0_distances_df,
        d1_distances=d1_distances_df
    )


def validate_parameters(d0: pd.DataFrame, d1: pd.DataFrame, x_target: int, 
                       p_set: int, delta: float) -> Tuple[bool, str]:
    """
    Validate input parameters for DRG function.
    
    Args:
        d0: Original dataset
        d1: Obfuscated dataset
        x_target: Target row index
        p_set: Number of features
        delta: Radius parameter
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check data dimensions
    if d0.shape != d1.shape:
        return False, f"Data dimensions mismatch: D0 {d0.shape} vs D1 {d1.shape}"
    
    # Check column names
    if list(d0.columns) != list(d1.columns):
        return False, "Column names mismatch between D0 and D1"
    
    # Check target index
    if x_target < 1 or x_target > len(d0):
        return False, f"Target index {x_target} out of range [1, {len(d0)}]"
    
    # Check complexity
    if p_set < 2 or p_set > d0.shape[1] - 1:
        return False, f"Complexity {p_set} out of range [2, {d0.shape[1] - 1}]"
    
    # Check radius
    if delta < 0 or delta > 1:
        return False, f"Radius {delta} out of range [0, 1]"
    
    return True, ""

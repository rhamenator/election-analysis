"""
Election Data Ingestion Module

This module handles loading, cleaning, and validating precinct-level election data.
Supports CSV files with standardized column structure and implements robust
data validation and normalization procedures.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectionDataIngester:
    """
    Handles ingestion and preprocessing of precinct-level election data.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.required_columns = self.config['data']['required_columns']
        self.validation_rules = self.config['data']['validation']
        self.missing_strategy = self.config['data']['missing_value_strategy']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if config file is missing."""
        return {
            'data': {
                'required_columns': [
                    'County', 'Precinct', 'Lat', 'Lon', 'Registered_Dem',
                    'Registered_Rep', 'Votes_Harris', 'Votes_Trump',
                    'Total_Votes', 'Turnout_Percent'
                ],
                'validation': {
                    'lat_range': [-90, 90],
                    'lon_range': [-180, 180],
                    'turnout_range': [0, 100],
                    'min_votes': 0
                },
                'missing_value_strategy': 'interpolate'
            }
        }
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file with election data.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Try different encodings if default fails
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any supported encoding")
                
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that required columns are present.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with validated columns
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select only required columns and reorder
        df = df[self.required_columns].copy()
        logger.info("Column validation passed")
        return df
    
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        logger.info("Validating and converting data types")
        
        # Define expected data types
        dtype_mapping = {
            'County': 'string',
            'Precinct': 'string',
            'Lat': 'float64',
            'Lon': 'float64',
            'Registered_Dem': 'int64',
            'Registered_Rep': 'int64',
            'Votes_Harris': 'int64',
            'Votes_Trump': 'int64',
            'Total_Votes': 'int64',
            'Turnout_Percent': 'float64'
        }
        
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                try:
                    if dtype == 'string':
                        df[col] = df[col].astype(str)
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'int64':
                        # Convert to float first to handle NaN, then to int
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Don't convert to int yet if there are NaN values
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        return df
    
    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data ranges and flag outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with range validation flags
        """
        logger.info("Validating data ranges")
        
        validation_issues = []
        
        # Validate latitude
        lat_issues = (df['Lat'] < self.validation_rules['lat_range'][0]) | \
                    (df['Lat'] > self.validation_rules['lat_range'][1])
        if lat_issues.any():
            validation_issues.append(f"Invalid latitude values: {lat_issues.sum()} rows")
            df.loc[lat_issues, 'Lat'] = np.nan
        
        # Validate longitude
        lon_issues = (df['Lon'] < self.validation_rules['lon_range'][0]) | \
                    (df['Lon'] > self.validation_rules['lon_range'][1])
        if lon_issues.any():
            validation_issues.append(f"Invalid longitude values: {lon_issues.sum()} rows")
            df.loc[lon_issues, 'Lon'] = np.nan
        
        # Validate turnout percentage
        turnout_issues = (df['Turnout_Percent'] < self.validation_rules['turnout_range'][0]) | \
                        (df['Turnout_Percent'] > self.validation_rules['turnout_range'][1])
        if turnout_issues.any():
            validation_issues.append(f"Invalid turnout values: {turnout_issues.sum()} rows")
            df.loc[turnout_issues, 'Turnout_Percent'] = np.nan
        
        # Validate vote counts (non-negative)
        vote_cols = ['Registered_Dem', 'Registered_Rep', 'Votes_Harris', 'Votes_Trump', 'Total_Votes']
        for col in vote_cols:
            negative_votes = df[col] < self.validation_rules['min_votes']
            if negative_votes.any():
                validation_issues.append(f"Negative {col} values: {negative_votes.sum()} rows")
                df.loc[negative_votes, col] = np.nan
        
        if validation_issues:
            logger.warning("Validation issues found:")
            for issue in validation_issues:
                logger.warning(f"  - {issue}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to configured strategy.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {self.missing_strategy}")
        
        missing_before = df.isnull().sum()
        
        if self.missing_strategy == 'drop':
            df = df.dropna()
            logger.info(f"Dropped {len(missing_before) - len(df)} rows with missing values")
            
        elif self.missing_strategy == 'interpolate':
            # Geographic interpolation for Lat/Lon within counties
            for county in df['County'].unique():
                county_mask = df['County'] == county
                county_data = df[county_mask]
                
                if county_data['Lat'].isnull().any():
                    df.loc[county_mask, 'Lat'] = county_data['Lat'].interpolate(method='linear')
                if county_data['Lon'].isnull().any():
                    df.loc[county_mask, 'Lon'] = county_data['Lon'].interpolate(method='linear')
            
            # Numerical interpolation for other columns
            numeric_cols = ['Registered_Dem', 'Registered_Rep', 'Votes_Harris', 
                          'Votes_Trump', 'Total_Votes', 'Turnout_Percent']
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method='linear')
            
        elif self.missing_strategy == 'mean':
            # Mean imputation by county
            for county in df['County'].unique():
                county_mask = df['County'] == county
                county_data = df[county_mask]
                
                for col in df.select_dtypes(include=[np.number]).columns:
                    if county_data[col].isnull().any():
                        mean_value = county_data[col].mean()
                        if not np.isnan(mean_value):
                            df.loc[county_mask & df[col].isnull(), col] = mean_value
        
        # Convert integer columns back to int (after handling NaN)
        int_cols = ['Registered_Dem', 'Registered_Rep', 'Votes_Harris', 'Votes_Trump', 'Total_Votes']
        for col in int_cols:
            if not df[col].isnull().any():
                df[col] = df[col].astype(int)
        
        missing_after = df.isnull().sum()
        logger.info("Missing values summary:")
        for col in df.columns:
            if missing_before[col] > 0:
                logger.info(f"  {col}: {missing_before[col]} → {missing_after[col]}")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        logger.info("Adding derived features")
        
        # Total registered voters
        df['Total_Registered'] = df['Registered_Dem'] + df['Registered_Rep']
        
        # Vote shares
        df['Harris_Share'] = np.where(df['Total_Votes'] > 0, 
                                    df['Votes_Harris'] / df['Total_Votes'], 0)
        df['Trump_Share'] = np.where(df['Total_Votes'] > 0, 
                                   df['Votes_Trump'] / df['Total_Votes'], 0)
        
        # Registration ratios
        df['Dem_Reg_Ratio'] = np.where(df['Total_Registered'] > 0,
                                     df['Registered_Dem'] / df['Total_Registered'], 0)
        df['Rep_Reg_Ratio'] = np.where(df['Total_Registered'] > 0,
                                     df['Registered_Rep'] / df['Total_Registered'], 0)
        
        # Turnout calculations
        df['Calculated_Turnout'] = np.where(df['Total_Registered'] > 0,
                                          df['Total_Votes'] / df['Total_Registered'] * 100, 0)
        
        # Flag discrepancies between reported and calculated turnout
        df['Turnout_Discrepancy'] = abs(df['Turnout_Percent'] - df['Calculated_Turnout'])
        
        # Competitive margin
        df['Competitive_Margin'] = abs(df['Harris_Share'] - df['Trump_Share'])
        
        logger.info(f"Added {len(df.columns) - len(self.required_columns)} derived features")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize text fields and standardize formats.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized data
        """
        logger.info("Normalizing data formats")
        
        # Normalize county and precinct names
        df['County'] = df['County'].str.strip().str.title()
        df['Precinct'] = df['Precinct'].str.strip()
        
        # Round numeric values to appropriate precision
        df['Lat'] = df['Lat'].round(6)
        df['Lon'] = df['Lon'].round(6)
        df['Turnout_Percent'] = df['Turnout_Percent'].round(2)
        
        # Add unique identifier for each precinct
        df['Precinct_ID'] = df['County'].astype(str) + "_" + df['Precinct'].astype(str)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_precincts': len(df),
            'total_counties': df['County'].nunique(),
            'total_votes': df['Total_Votes'].sum(),
            'avg_turnout': df['Turnout_Percent'].mean(),
            'median_turnout': df['Turnout_Percent'].median(),
            'harris_total': df['Votes_Harris'].sum(),
            'trump_total': df['Votes_Trump'].sum(),
            'data_quality': {
                'missing_coordinates': df[['Lat', 'Lon']].isnull().any(axis=1).sum(),
                'zero_votes': (df['Total_Votes'] == 0).sum(),
                'high_turnout': (df['Turnout_Percent'] > 100).sum(),
                'duplicate_precincts': df['Precinct_ID'].duplicated().sum()
            }
        }
        
        return summary
    
    def process_file(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete processing pipeline for election data file.
        
        Args:
            file_path: Path to election data CSV file
            
        Returns:
            Tuple of (processed DataFrame, summary statistics)
        """
        logger.info(f"Starting data processing pipeline for {file_path}")
        
        # Load data
        df = self.load_csv(file_path)
        
        # Validation and cleaning pipeline
        df = self.validate_columns(df)
        df = self.validate_data_types(df)
        df = self.validate_ranges(df)
        df = self.handle_missing_values(df)
        df = self.normalize_data(df)
        df = self.add_derived_features(df)
        
        # Generate summary
        summary = self.get_data_summary(df)
        
        logger.info("Data processing pipeline completed successfully")
        logger.info(f"Final dataset: {summary['total_precincts']} precincts, "
                   f"{summary['total_counties']} counties")
        
        return df, summary


def main():
    """Example usage of the data ingestion module."""
    # Initialize ingester
    ingester = ElectionDataIngester()
    
    # Process example file (would need actual data file)
    try:
        df, summary = ingester.process_file("example_election_data.csv")
        print("Data processing completed successfully!")
        print(f"Processed {summary['total_precincts']} precincts")
        print(f"Data quality issues: {summary['data_quality']}")
    except FileNotFoundError:
        print("Example data file not found. Please provide actual election data.")
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'County': ['County A', 'County A', 'County B'],
            'Precinct': ['Precinct 1', 'Precinct 2', 'Precinct 1'],
            'Lat': [40.7128, 40.7589, 34.0522],
            'Lon': [-74.0060, -73.9851, -118.2437],
            'Registered_Dem': [1000, 1500, 800],
            'Registered_Rep': [1200, 1000, 1200],
            'Votes_Harris': [450, 700, 380],
            'Votes_Trump': [520, 400, 550],
            'Total_Votes': [1000, 1150, 950],
            'Turnout_Percent': [45.5, 57.5, 47.5]
        })
        print("Created sample dataset:")
        print(sample_data.head())


if __name__ == "__main__":
    main()
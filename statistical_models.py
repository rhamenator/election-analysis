"""
Statistical Models Module

This module implements advanced statistical analyses for election anomaly detection,
including Shpilkin-style turnout analysis, spatial autocorrelation (Moran's I),
and entropy-based clustering detection following Klimek methodologies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, spatial
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

try:
    from libpysal.weights import Queen, Rook, KNN
    from esda.moran import Moran, Moran_Local
    SPATIAL_AVAILABLE = True
except ImportError:
    warnings.warn("Spatial analysis libraries not available. Install libpysal and esda for full functionality.")
    SPATIAL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShpilkinAnalyzer:
    """
    Implements Shpilkin-style turnout vs vote share analysis for anomaly detection.
    
    The Shpilkin method identifies statistical anomalies by analyzing the relationship
    between voter turnout and vote share, looking for deviations from expected patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration parameters."""
        self.config = config or {
            'polynomial_degree': 3,
            'confidence_interval': 0.95,
            'bin_size': 0.02,
            'min_precincts_per_bin': 5
        }
        
    def prepare_turnout_data(self, df: pd.DataFrame, 
                           candidate_column: str = 'Harris_Share') -> pd.DataFrame:
        """
        Prepare data for turnout analysis.
        
        Args:
            df: DataFrame with election data
            candidate_column: Column name for candidate vote share
            
        Returns:
            DataFrame ready for analysis
        """
        analysis_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['Turnout_Percent', candidate_column, 'Total_Votes']
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out extreme outliers and invalid data
        analysis_df = analysis_df[
            (analysis_df['Turnout_Percent'] > 0) & 
            (analysis_df['Turnout_Percent'] <= 100) &
            (analysis_df[candidate_column] >= 0) &
            (analysis_df[candidate_column] <= 1) &
            (analysis_df['Total_Votes'] > 0)
        ].copy()
        
        logger.info(f"Prepared {len(analysis_df)} precincts for Shpilkin analysis")
        return analysis_df
    
    def create_turnout_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create turnout bins for analysis.
        
        Args:
            df: Prepared election data
            
        Returns:
            DataFrame with turnout bins added
        """
        bin_size = self.config['bin_size']
        
        # Create turnout bins
        max_turnout = df['Turnout_Percent'].max()
        bins = np.arange(0, max_turnout + bin_size, bin_size * 100)  # Convert to percentage
        
        df['Turnout_Bin'] = pd.cut(df['Turnout_Percent'], 
                                  bins=bins, 
                                  include_lowest=True,
                                  labels=False)
        
        # Calculate bin centers
        bin_centers = []
        for i in range(len(bins) - 1):
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
        
        df['Turnout_Bin_Center'] = df['Turnout_Bin'].map(
            dict(enumerate(bin_centers))
        )
        
        return df
    
    def fit_expected_curve(self, df: pd.DataFrame, 
                          candidate_column: str = 'Harris_Share') -> Dict:
        """
        Fit expected vote share curve using polynomial regression.
        
        Args:
            df: Binned election data
            candidate_column: Column for candidate vote share
            
        Returns:
            Dictionary with fitted model and statistics
        """
        # Group by turnout bins and calculate statistics
        bin_stats = df.groupby('Turnout_Bin_Center').agg({
            candidate_column: ['mean', 'std', 'count'],
            'Total_Votes': 'sum'
        }).reset_index()
        
        bin_stats.columns = ['Turnout', 'Vote_Share_Mean', 'Vote_Share_Std', 
                           'Precinct_Count', 'Total_Votes']
        
        # Filter bins with sufficient data
        min_precincts = self.config['min_precincts_per_bin']
        bin_stats = bin_stats[bin_stats['Precinct_Count'] >= min_precincts]
        
        if len(bin_stats) < 3:
            raise ValueError(f"Insufficient data for curve fitting. "
                           f"Need at least 3 bins with {min_precincts}+ precincts each.")
        
        # Fit polynomial regression
        degree = self.config['polynomial_degree']
        X = bin_stats['Turnout'].values.reshape(-1, 1)
        y = bin_stats['Vote_Share_Mean'].values
        
        # Weight by number of precincts and total votes
        weights = np.sqrt(bin_stats['Precinct_Count'] * bin_stats['Total_Votes'])
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y, sample_weight=weights)
        
        # Calculate predictions and confidence intervals
        y_pred = model.predict(X_poly)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # Calculate confidence intervals
        confidence_level = self.config['confidence_interval']
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(bin_stats) - degree - 1)
        margin_error = t_value * np.sqrt(mse)
        
        results = {
            'model': model,
            'poly_features': poly_features,
            'bin_stats': bin_stats,
            'predictions': y_pred,
            'confidence_interval': margin_error,
            'r_squared': model.score(X_poly, y, sample_weight=weights),
            'mse': mse,
            'turnout_range': (X.min(), X.max())
        }
        
        logger.info(f"Fitted polynomial curve (degree {degree}) with R² = {results['r_squared']:.3f}")
        return results
    
    def detect_anomalies(self, df: pd.DataFrame, curve_results: Dict,
                        candidate_column: str = 'Harris_Share') -> pd.DataFrame:
        """
        Detect anomalies based on deviations from expected curve.
        
        Args:
            df: Original election data
            curve_results: Results from curve fitting
            candidate_column: Column for candidate vote share
            
        Returns:
            DataFrame with anomaly scores added
        """
        model = curve_results['model']
        poly_features = curve_results['poly_features']
        confidence_interval = curve_results['confidence_interval']
        
        # Predict expected vote shares for all precincts
        X = df['Turnout_Percent'].values.reshape(-1, 1)
        X_poly = poly_features.transform(X)
        expected_share = model.predict(X_poly)
        
        # Calculate deviations
        actual_share = df[candidate_column].values
        deviation = actual_share - expected_share
        
        # Normalize by confidence interval to get anomaly scores
        anomaly_score = np.abs(deviation) / confidence_interval
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['Expected_Vote_Share'] = expected_share
        result_df['Vote_Share_Deviation'] = deviation
        result_df['Shpilkin_Anomaly_Score'] = anomaly_score
        
        # Flag significant anomalies
        threshold = 2.0  # 2 standard deviations
        result_df['Shpilkin_Anomaly_Flag'] = anomaly_score > threshold
        
        logger.info(f"Detected {result_df['Shpilkin_Anomaly_Flag'].sum()} Shpilkin anomalies "
                   f"({result_df['Shpilkin_Anomaly_Flag'].mean()*100:.1f}% of precincts)")
        
        return result_df
    
    def analyze_excess_votes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate excess votes at high turnout levels.
        
        Args:
            df: DataFrame with anomaly analysis
            
        Returns:
            DataFrame with excess vote calculations
        """
        # Define high turnout threshold (typically >80%)
        high_turnout_threshold = 80
        
        result_df = df.copy()
        
        # Calculate excess votes for high turnout precincts
        high_turnout_mask = df['Turnout_Percent'] > high_turnout_threshold
        
        if high_turnout_mask.any():
            # Estimate expected votes based on lower turnout precincts
            normal_turnout = df[~high_turnout_mask]
            if len(normal_turnout) > 0:
                avg_normal_turnout = normal_turnout['Turnout_Percent'].mean()
                
                # Calculate excess votes
                result_df['Expected_Votes_Normal'] = (
                    result_df['Total_Registered'] * avg_normal_turnout / 100
                )
                result_df['Excess_Votes'] = np.maximum(
                    0, result_df['Total_Votes'] - result_df['Expected_Votes_Normal']
                )
                
                # Calculate excess vote percentage
                result_df['Excess_Vote_Percent'] = np.where(
                    result_df['Expected_Votes_Normal'] > 0,
                    result_df['Excess_Votes'] / result_df['Expected_Votes_Normal'] * 100,
                    0
                )
            else:
                result_df['Excess_Votes'] = 0
                result_df['Excess_Vote_Percent'] = 0
        else:
            result_df['Excess_Votes'] = 0
            result_df['Excess_Vote_Percent'] = 0
        
        return result_df


class SpatialAnalyzer:
    """
    Implements spatial autocorrelation analysis using Moran's I statistic.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration parameters."""
        if not SPATIAL_AVAILABLE:
            raise ImportError("Spatial analysis libraries not available. "
                            "Install libpysal and esda for spatial analysis.")
        
        self.config = config or {
            'weights_type': 'queen',
            'k_neighbors': 8,
            'significance_level': 0.05
        }
    
    def create_spatial_weights(self, df: pd.DataFrame) -> object:
        """
        Create spatial weights matrix from coordinates.
        
        Args:
            df: DataFrame with Lat/Lon coordinates
            
        Returns:
            Spatial weights object
        """
        coords = df[['Lon', 'Lat']].values
        
        weights_type = self.config['weights_type']
        
        if weights_type == 'knn':
            k = self.config['k_neighbors']
            weights = KNN.from_array(coords, k=k)
        else:
            # For queen/rook, we'll use KNN as approximation since we don't have polygons
            # In practice, you'd want actual precinct boundary polygons
            logger.warning("Using KNN approximation for spatial weights. "
                          "Consider using actual precinct polygons for better results.")
            weights = KNN.from_array(coords, k=self.config['k_neighbors'])
        
        weights.transform = 'r'  # Row-standardize
        logger.info(f"Created spatial weights matrix with {weights.n} observations")
        return weights
    
    def calculate_global_moran(self, df: pd.DataFrame, 
                              variable: str, weights: object) -> Dict:
        """
        Calculate Global Moran's I statistic.
        
        Args:
            df: DataFrame with analysis data
            variable: Variable to analyze for spatial autocorrelation
            weights: Spatial weights matrix
            
        Returns:
            Dictionary with Moran's I results
        """
        values = df[variable].values
        
        # Handle missing values
        valid_mask = ~np.isnan(values)
        if not valid_mask.all():
            logger.warning(f"Removing {(~valid_mask).sum()} observations with missing {variable}")
            values = values[valid_mask]
            # Would need to subset weights matrix accordingly
        
        moran = Moran(values, weights)
        
        results = {
            'I': moran.I,
            'expected_I': moran.EI,
            'variance': moran.VI_norm,
            'z_score': moran.z_norm,
            'p_value': moran.p_norm,
            'significant': moran.p_norm < self.config['significance_level']
        }
        
        logger.info(f"Global Moran's I for {variable}: {results['I']:.4f} "
                   f"(p-value: {results['p_value']:.4f})")
        
        return results
    
    def calculate_local_moran(self, df: pd.DataFrame, 
                             variable: str, weights: object) -> pd.DataFrame:
        """
        Calculate Local Moran's I (LISA) statistics.
        
        Args:
            df: DataFrame with analysis data
            variable: Variable to analyze
            weights: Spatial weights matrix
            
        Returns:
            DataFrame with local Moran's I results
        """
        values = df[variable].values
        
        lisa = Moran_Local(values, weights)
        
        result_df = df.copy()
        result_df['Local_Moran_I'] = lisa.Is
        result_df['Local_Moran_p'] = lisa.p_sim
        result_df['Local_Moran_z'] = lisa.z_sim
        result_df['Local_Moran_Quadrant'] = lisa.q
        
        # Flag significant local clusters
        significance_level = self.config['significance_level']
        result_df['Spatial_Cluster_Flag'] = lisa.p_sim < significance_level
        
        # Interpret quadrants
        quadrant_labels = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        result_df['Cluster_Type'] = result_df['Local_Moran_Quadrant'].map(quadrant_labels)
        
        logger.info(f"Local Moran analysis complete. "
                   f"Found {result_df['Spatial_Cluster_Flag'].sum()} significant spatial clusters")
        
        return result_df


class EntropyAnalyzer:
    """
    Implements Klimek-style entropy analysis for detecting statistical fingerprints.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize entropy analyzer."""
        self.config = config or {
            'bin_size': 0.01,
            'min_digit_frequency': 0.05
        }
    
    def calculate_digit_entropy(self, values: np.ndarray, 
                               digit_position: int = -1) -> Dict:
        """
        Calculate entropy of digit distributions (Benford's Law analysis).
        
        Args:
            values: Array of numeric values
            digit_position: Position of digit to analyze (-1 for last digit)
            
        Returns:
            Dictionary with entropy statistics
        """
        # Extract digits
        if digit_position == -1:
            # Last digit analysis
            digits = np.array([int(str(int(v))[-1]) if v > 0 else 0 for v in values])
        else:
            # Specific position (0-based from left)
            digits = np.array([
                int(str(int(v))[digit_position]) 
                if len(str(int(v))) > digit_position and v > 0 else 0 
                for v in values
            ])
        
        # Calculate frequency distribution
        digit_counts = np.bincount(digits, minlength=10)
        digit_freq = digit_counts / len(digits)
        
        # Calculate entropy
        # Remove zero frequencies to avoid log(0)
        nonzero_freq = digit_freq[digit_freq > 0]
        entropy = -np.sum(nonzero_freq * np.log2(nonzero_freq))
        
        # Maximum possible entropy (uniform distribution)
        max_entropy = np.log2(10)  # log2 of number of possible digits
        
        # Normalize entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        results = {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'digit_frequencies': digit_freq,
            'digit_counts': digit_counts,
            'uniformity_score': 1 - np.std(digit_freq[digit_freq > 0])
        }
        
        return results
    
    def analyze_vote_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze vote count patterns for statistical fingerprints.
        
        Args:
            df: DataFrame with vote data
            
        Returns:
            DataFrame with entropy analysis results
        """
        result_df = df.copy()
        
        vote_columns = ['Votes_Harris', 'Votes_Trump', 'Total_Votes']
        
        for col in vote_columns:
            if col in df.columns:
                # Last digit entropy
                entropy_results = self.calculate_digit_entropy(df[col].values, -1)
                result_df[f'{col}_Last_Digit_Entropy'] = entropy_results['normalized_entropy']
                
                # First digit entropy (if applicable)
                if df[col].max() >= 10:
                    first_digit_results = self.calculate_digit_entropy(df[col].values, 0)
                    result_df[f'{col}_First_Digit_Entropy'] = first_digit_results['normalized_entropy']
        
        return result_df
    
    def detect_round_number_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect preference for round numbers in vote counts.
        
        Args:
            df: DataFrame with vote data
            
        Returns:
            DataFrame with round number analysis
        """
        result_df = df.copy()
        
        vote_columns = ['Votes_Harris', 'Votes_Trump', 'Total_Votes']
        
        for col in vote_columns:
            if col in df.columns:
                values = df[col].values
                
                # Check for multiples of 5, 10, 25, 50, 100
                round_checks = {
                    'mod_5': values % 5 == 0,
                    'mod_10': values % 10 == 0,
                    'mod_25': values % 25 == 0,
                    'mod_50': values % 50 == 0,
                    'mod_100': values % 100 == 0
                }
                
                for check_name, check_result in round_checks.items():
                    result_df[f'{col}_{check_name}'] = check_result
                
                # Calculate overall "roundness" score
                roundness_score = (
                    check_result['mod_5'].astype(int) +
                    check_result['mod_10'].astype(int) * 2 +
                    check_result['mod_25'].astype(int) * 3 +
                    check_result['mod_50'].astype(int) * 4 +
                    check_result['mod_100'].astype(int) * 5
                ) / 15  # Normalize to 0-1
                
                result_df[f'{col}_Roundness_Score'] = roundness_score
        
        return result_df


class StatisticalAnomalyDetector:
    """
    Main class that combines all statistical analysis methods.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.config = config.get('statistics', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.config = {}
        
        # Initialize analyzers
        self.shpilkin = ShpilkinAnalyzer(self.config.get('shpilkin'))
        
        if SPATIAL_AVAILABLE:
            self.spatial = SpatialAnalyzer(self.config.get('spatial'))
        else:
            self.spatial = None
            logger.warning("Spatial analysis not available")
        
        self.entropy = EntropyAnalyzer(self.config.get('entropy'))
        
    def run_full_analysis(self, df: pd.DataFrame, 
                         candidate_column: str = 'Harris_Share') -> pd.DataFrame:
        """
        Run complete statistical anomaly detection pipeline.
        
        Args:
            df: Input DataFrame with election data
            candidate_column: Column name for candidate vote share
            
        Returns:
            DataFrame with all anomaly detection results
        """
        logger.info("Starting comprehensive statistical analysis")
        
        result_df = df.copy()
        
        # 1. Shpilkin Analysis
        logger.info("Running Shpilkin turnout analysis...")
        shpilkin_data = self.shpilkin.prepare_turnout_data(df, candidate_column)
        shpilkin_data = self.shpilkin.create_turnout_bins(shpilkin_data)
        curve_results = self.shpilkin.fit_expected_curve(shpilkin_data, candidate_column)
        result_df = self.shpilkin.detect_anomalies(result_df, curve_results, candidate_column)
        result_df = self.shpilkin.analyze_excess_votes(result_df)
        
        # 2. Spatial Analysis (if available)
        if self.spatial and 'Lat' in df.columns and 'Lon' in df.columns:
            logger.info("Running spatial autocorrelation analysis...")
            try:
                weights = self.spatial.create_spatial_weights(df)
                
                # Global Moran's I for vote shares
                global_moran = self.spatial.calculate_global_moran(
                    df, candidate_column, weights
                )
                
                # Local Moran's I
                result_df = self.spatial.calculate_local_moran(
                    result_df, candidate_column, weights
                )
                
                # Store global results as attributes
                result_df.attrs['global_moran'] = global_moran
                
            except Exception as e:
                logger.error(f"Spatial analysis failed: {e}")
        
        # 3. Entropy Analysis
        logger.info("Running entropy and digit analysis...")
        result_df = self.entropy.analyze_vote_patterns(result_df)
        result_df = self.entropy.detect_round_number_preference(result_df)
        
        # 4. Composite Anomaly Score
        logger.info("Calculating composite anomaly scores...")
        result_df = self._calculate_composite_score(result_df)
        
        logger.info("Statistical analysis pipeline completed")
        return result_df
    
    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite anomaly score from all analyses."""
        scores = []
        
        # Shpilkin score (normalized)
        if 'Shpilkin_Anomaly_Score' in df.columns:
            shpilkin_norm = np.clip(df['Shpilkin_Anomaly_Score'] / 5, 0, 1)
            scores.append(shpilkin_norm)
        
        # Spatial clustering score
        if 'Spatial_Cluster_Flag' in df.columns:
            spatial_score = df['Spatial_Cluster_Flag'].astype(float)
            scores.append(spatial_score)
        
        # Entropy deviation scores
        entropy_cols = [col for col in df.columns if 'Entropy' in col]
        if entropy_cols:
            # Lower entropy = higher anomaly for digit analysis
            avg_entropy = df[entropy_cols].mean(axis=1)
            entropy_score = 1 - avg_entropy  # Invert so low entropy = high anomaly
            scores.append(entropy_score)
        
        # Round number preference
        roundness_cols = [col for col in df.columns if 'Roundness_Score' in col]
        if roundness_cols:
            avg_roundness = df[roundness_cols].mean(axis=1)
            scores.append(avg_roundness)
        
        if scores:
            # Combine scores with equal weighting
            composite_score = np.mean(scores, axis=0)
            df['Composite_Anomaly_Score'] = composite_score
            
            # Flag high composite anomalies
            threshold = self.config.get('anomaly_threshold', 0.7)
            df['Composite_Anomaly_Flag'] = composite_score > threshold
        
        return df


def main():
    """Example usage of statistical analysis."""
    # Create sample data
    np.random.seed(42)
    n_precincts = 1000
    
    sample_data = pd.DataFrame({
        'County': np.random.choice(['County A', 'County B', 'County C'], n_precincts),
        'Precinct': [f'Precinct {i}' for i in range(n_precincts)],
        'Lat': np.random.uniform(40, 42, n_precincts),
        'Lon': np.random.uniform(-75, -73, n_precincts),
        'Total_Registered': np.random.randint(500, 3000, n_precincts),
        'Turnout_Percent': np.random.beta(5, 3) * 100,  # Realistic turnout distribution
    })
    
    # Add vote counts based on turnout
    sample_data['Total_Votes'] = (
        sample_data['Total_Registered'] * sample_data['Turnout_Percent'] / 100
    ).astype(int)
    
    # Add candidate vote shares with some correlation to turnout
    base_harris_share = 0.48 + np.random.normal(0, 0.1, n_precincts)
    turnout_effect = (sample_data['Turnout_Percent'] - 50) * 0.001
    sample_data['Harris_Share'] = np.clip(base_harris_share + turnout_effect, 0, 1)
    sample_data['Votes_Harris'] = (sample_data['Total_Votes'] * sample_data['Harris_Share']).astype(int)
    sample_data['Votes_Trump'] = sample_data['Total_Votes'] - sample_data['Votes_Harris']
    
    # Run analysis
    detector = StatisticalAnomalyDetector()
    results = detector.run_full_analysis(sample_data)
    
    print(f"Analysis completed on {len(results)} precincts")
    if 'Composite_Anomaly_Score' in results.columns:
        anomaly_count = results['Composite_Anomaly_Flag'].sum()
        print(f"Found {anomaly_count} composite anomalies ({anomaly_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
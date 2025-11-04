"""
Machine Learning Anomaly Detection Module

This module implements unsupervised machine learning approaches for detecting
statistical outliers in election data, including Isolation Forest, DBSCAN clustering,
and SHAP-based interpretability analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import joblib
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    warnings.warn("SHAP not available. Install shap for interpretability analysis.")
    SHAP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for machine learning anomaly detection.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []
        self.scaler = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for anomaly detection.
        
        Args:
            df: Input DataFrame with election data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features for ML anomaly detection")
        
        features_df = df.copy()
        
        # Basic electoral features
        if all(col in df.columns for col in ['Votes_Harris', 'Votes_Trump', 'Total_Votes']):
            features_df['Vote_Margin'] = abs(features_df['Votes_Harris'] - features_df['Votes_Trump'])
            features_df['Vote_Margin_Pct'] = features_df['Vote_Margin'] / features_df['Total_Votes']
            features_df['Harris_Trump_Ratio'] = np.where(
                features_df['Votes_Trump'] > 0,
                features_df['Votes_Harris'] / features_df['Votes_Trump'],
                features_df['Votes_Harris']
            )
        
        # Registration vs voting patterns
        if all(col in df.columns for col in ['Registered_Dem', 'Registered_Rep', 'Total_Votes']):
            features_df['Reg_Voting_Efficiency'] = features_df['Total_Votes'] / (
                features_df['Registered_Dem'] + features_df['Registered_Rep']
            )
            
            # Party loyalty metrics
            features_df['Dem_Loyalty'] = np.where(
                features_df['Registered_Dem'] > 0,
                features_df['Votes_Harris'] / features_df['Registered_Dem'],
                0
            )
            features_df['Rep_Loyalty'] = np.where(
                features_df['Registered_Rep'] > 0,
                features_df['Votes_Trump'] / features_df['Registered_Rep'],
                0
            )
        
        # Turnout-based features
        if 'Turnout_Percent' in df.columns:
            # Turnout categories
            features_df['Low_Turnout'] = (features_df['Turnout_Percent'] < 40).astype(int)
            features_df['Medium_Turnout'] = ((features_df['Turnout_Percent'] >= 40) & 
                                           (features_df['Turnout_Percent'] < 70)).astype(int)
            features_df['High_Turnout'] = (features_df['Turnout_Percent'] >= 70).astype(int)
            features_df['Very_High_Turnout'] = (features_df['Turnout_Percent'] >= 90).astype(int)
            
            # Turnout squared (non-linear effects)\n            features_df['Turnout_Squared'] = features_df['Turnout_Percent'] ** 2
        
        # Vote count characteristics
        if 'Total_Votes' in df.columns:
            features_df['Log_Total_Votes'] = np.log1p(features_df['Total_Votes'])
            features_df['Vote_Density'] = features_df['Total_Votes'] / features_df.groupby('County')['Total_Votes'].transform('mean')
            
            # Round number detection
            features_df['Votes_End_0'] = (features_df['Total_Votes'] % 10 == 0).astype(int)
            features_df['Votes_End_5'] = (features_df['Total_Votes'] % 5 == 0).astype(int)
            features_df['Votes_Divisible_25'] = (features_df['Total_Votes'] % 25 == 0).astype(int)
        
        # Geographic features (if available)
        if all(col in df.columns for col in ['Lat', 'Lon']):
            # Distance from county centroid
            county_centroids = features_df.groupby('County')[['Lat', 'Lon']].mean()
            features_df = features_df.merge(
                county_centroids.add_suffix('_County_Centroid'), 
                left_on='County', 
                right_index=True
            )
            
            features_df['Distance_From_County_Center'] = np.sqrt(
                (features_df['Lat'] - features_df['Lat_County_Centroid'])**2 + 
                (features_df['Lon'] - features_df['Lon_County_Centroid'])**2
            )
        
        # County-level aggregations
        county_stats = features_df.groupby('County').agg({
            'Total_Votes': ['mean', 'std'],
            'Turnout_Percent': ['mean', 'std'],
            'Harris_Share': ['mean', 'std'] if 'Harris_Share' in features_df.columns else 'count'
        }).round(4)
        
        county_stats.columns = ['_'.join(col).strip() for col in county_stats.columns]
        county_stats = county_stats.add_prefix('County_')
        
        features_df = features_df.merge(county_stats, left_on='County', right_index=True)
        
        # Deviation from county norms
        if 'Total_Votes_mean' in county_stats.columns:
            features_df['Votes_County_Deviation'] = (
                features_df['Total_Votes'] - features_df['County_Total_Votes_mean']
            ) / (features_df['County_Total_Votes_std'] + 1e-8)
            
            features_df['Turnout_County_Deviation'] = (
                features_df['Turnout_Percent'] - features_df['County_Turnout_Percent_mean']
            ) / (features_df['County_Turnout_Percent_std'] + 1e-8)
        
        # Store feature column names (exclude identifiers and targets)
        exclude_columns = ['County', 'Precinct', 'Precinct_ID', 'Lat', 'Lon']
        self.feature_columns = [col for col in features_df.columns 
                               if col not in exclude_columns 
                               and not col.endswith('_Flag') 
                               and not col.endswith('_Score')]
        
        logger.info(f"Created {len(self.feature_columns)} features for ML analysis")
        return features_df
    
    def prepare_features_for_ml(self, df: pd.DataFrame, 
                               feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare feature matrix for ML algorithms.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to use as features
            
        Returns:
            Scaled feature matrix
        """
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # Select features and handle missing values
        X = df[feature_columns].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Scale features
        if self.scaler is None:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        logger.info(f"Prepared feature matrix: {X_scaled.shape}")
        return X_scaled


class IsolationForestDetector:
    """
    Isolation Forest anomaly detection for election data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Isolation Forest detector."""
        self.config = config or {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'max_samples': 'auto'
        }
        self.model = None
        self.feature_importance = None
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        Fit Isolation Forest model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Isolation Forest model")
        
        self.model = IsolationForest(
            contamination=self.config['contamination'],
            n_estimators=self.config['n_estimators'],
            random_state=self.config['random_state'],
            max_samples=self.config['max_samples'],
            n_jobs=-1
        )
        
        self.model.fit(X)
        logger.info(f"Trained Isolation Forest with {self.config['n_estimators']} trees")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get anomaly scores (negative for anomalies)
        anomaly_scores = self.model.decision_function(X)
        
        # Get binary predictions (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(X)
        anomaly_flags = (predictions == -1).astype(int)
        
        # Convert scores to positive (higher = more anomalous)
        anomaly_scores = -anomaly_scores
        
        logger.info(f"Detected {anomaly_flags.sum()} anomalies "
                   f"({anomaly_flags.mean()*100:.1f}% of data)")
        
        return anomaly_flags, anomaly_scores
    
    def get_feature_importance(self, X: np.ndarray, 
                             feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance using permutation method.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calculating importance")
        
        logger.info("Calculating feature importance")
        
        # Get baseline scores
        baseline_scores = self.model.decision_function(X)
        baseline_mean = np.mean(baseline_scores)
        
        importance_scores = []
        
        for i, feature_name in enumerate(feature_names):
            # Create permuted version
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            
            # Calculate new scores
            perm_scores = self.model.decision_function(X_perm)
            perm_mean = np.mean(perm_scores)
            
            # Importance is the change in mean score
            importance = abs(baseline_mean - perm_mean)
            importance_scores.append(importance)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        logger.info("Feature importance calculation completed")
        
        return importance_df


class DBSCANDetector:
    """
    DBSCAN clustering for anomaly detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize DBSCAN detector."""
        self.config = config or {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean'
        }
        self.model = None
        self.cluster_stats = None
        
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit DBSCAN and predict clusters.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (cluster_labels, anomaly_flags)
        """
        logger.info("Running DBSCAN clustering")
        
        self.model = DBSCAN(
            eps=self.config['eps'],
            min_samples=self.config['min_samples'],
            metric=self.config['metric'],
            n_jobs=-1
        )
        
        cluster_labels = self.model.fit_predict(X)
        
        # Points with label -1 are considered noise/anomalies
        anomaly_flags = (cluster_labels == -1).astype(int)
        
        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        self.cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(cluster_labels),
            'silhouette_score': silhouette_score(X, cluster_labels) if n_clusters > 1 else 0
        }
        
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points "
                   f"({self.cluster_stats['noise_ratio']*100:.1f}% anomalies)")
        
        return cluster_labels, anomaly_flags
    
    def get_cluster_analysis(self, X: np.ndarray, 
                           cluster_labels: np.ndarray,
                           feature_names: List[str]) -> Dict:
        """
        Analyze cluster characteristics.
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster assignments
            feature_names: List of feature names
            
        Returns:
            Dictionary with cluster analysis
        """
        analysis = {}
        
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
                
            cluster_mask = cluster_labels == label
            cluster_data = X[cluster_mask]
            
            analysis[f'cluster_{label}'] = {
                'size': np.sum(cluster_mask),
                'centroid': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0),
                'feature_means': dict(zip(feature_names, np.mean(cluster_data, axis=0)))
            }
        
        return analysis


class SHAPExplainer:
    """
    SHAP-based interpretability for anomaly detection models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install shap for interpretability analysis.")
        
        self.config = config or {
            'max_display': 10,
            'sample_size': 1000
        }
        self.explainer = None
        self.shap_values = None
        
    def explain_isolation_forest(self, model: IsolationForest, 
                               X: np.ndarray, 
                               feature_names: List[str]) -> np.ndarray:
        """
        Create SHAP explanations for Isolation Forest predictions.
        
        Args:
            model: Trained Isolation Forest model
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            SHAP values array
        """
        logger.info("Creating SHAP explanations for Isolation Forest")
        
        # Sample data if too large
        if len(X) > self.config['sample_size']:
            indices = np.random.choice(len(X), self.config['sample_size'], replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create explainer
        self.explainer = shap.Explainer(model.decision_function, X_sample)
        
        # Calculate SHAP values
        self.shap_values = self.explainer(X_sample)
        
        logger.info(f"Generated SHAP explanations for {len(X_sample)} samples")
        return self.shap_values
    
    def plot_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_* method first")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, max_display=self.config['max_display'],
                         show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(self, sample_idx: int, save_path: Optional[str] = None) -> None:
        """
        Create SHAP waterfall plot for specific sample.
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_* method first")
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(self.shap_values[sample_idx], max_display=self.config['max_display'],
                           show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance_df(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with SHAP-based feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_* method first")
        
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(self.shap_values.values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance_scores
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df


class MLAnomalyDetector:
    """
    Main class that orchestrates all ML anomaly detection methods.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ML anomaly detector."""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.config = config.get('ml', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.config = {}
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.isolation_forest = IsolationForestDetector(self.config.get('isolation_forest'))
        self.dbscan = DBSCANDetector(self.config.get('dbscan'))
        
        if SHAP_AVAILABLE:
            self.shap_explainer = SHAPExplainer(self.config.get('shap'))
        else:
            self.shap_explainer = None
            logger.warning("SHAP explainer not available")
        
        # Store results
        self.feature_names = []
        self.models_fitted = False
        
    def fit_models(self, df: pd.DataFrame) -> 'MLAnomalyDetector':
        """
        Fit all ML models on the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ML anomaly detection models")
        
        # Create features
        features_df = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_features_for_ml(features_df)
        self.feature_names = self.feature_engineer.feature_columns
        
        # Fit models
        self.isolation_forest.fit(X)
        
        self.models_fitted = True
        logger.info("All ML models fitted successfully")
        
        return self
    
    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using all fitted models.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ML anomaly predictions
        """
        if not self.models_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        logger.info("Predicting anomalies with ML models")
        
        # Prepare features
        features_df = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_features_for_ml(features_df, self.feature_names)
        
        result_df = df.copy()
        
        # Isolation Forest predictions
        if_anomalies, if_scores = self.isolation_forest.predict(X)
        result_df['IF_Anomaly_Flag'] = if_anomalies
        result_df['IF_Anomaly_Score'] = if_scores
        
        # DBSCAN predictions
        cluster_labels, dbscan_anomalies = self.dbscan.fit_predict(X)
        result_df['DBSCAN_Cluster'] = cluster_labels
        result_df['DBSCAN_Anomaly_Flag'] = dbscan_anomalies
        
        # Composite ML score
        result_df['ML_Composite_Score'] = (
            (result_df['IF_Anomaly_Score'] / result_df['IF_Anomaly_Score'].max()) * 0.7 +
            result_df['DBSCAN_Anomaly_Flag'] * 0.3
        )
        
        # Flag high ML anomalies
        ml_threshold = 0.7
        result_df['ML_Anomaly_Flag'] = result_df['ML_Composite_Score'] > ml_threshold
        
        logger.info(f"ML anomaly detection completed. "
                   f"Found {result_df['ML_Anomaly_Flag'].sum()} ML anomalies")
        
        return result_df
    
    def explain_predictions(self, df: pd.DataFrame, 
                          save_plots: bool = False) -> Dict[str, Any]:
        """
        Generate explanations for ML predictions.
        
        Args:
            df: Input DataFrame
            save_plots: Whether to save explanation plots
            
        Returns:
            Dictionary with explanation results
        """
        if not self.models_fitted:
            raise ValueError("Models must be fitted before explanation")
        
        logger.info("Generating ML model explanations")
        
        # Prepare features
        features_df = self.feature_engineer.create_features(df)
        X = self.feature_engineer.prepare_features_for_ml(features_df, self.feature_names)
        
        explanations = {}
        
        # Feature importance from Isolation Forest
        if_importance = self.isolation_forest.get_feature_importance(X, self.feature_names)
        explanations['isolation_forest_importance'] = if_importance
        
        # DBSCAN cluster analysis
        cluster_labels, _ = self.dbscan.fit_predict(X)
        cluster_analysis = self.dbscan.get_cluster_analysis(X, cluster_labels, self.feature_names)
        explanations['dbscan_clusters'] = cluster_analysis
        
        # SHAP explanations (if available)
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer.explain_isolation_forest(
                    self.isolation_forest.model, X, self.feature_names
                )
                shap_importance = self.shap_explainer.get_feature_importance_df(self.feature_names)
                explanations['shap_importance'] = shap_importance
                
                if save_plots:
                    self.shap_explainer.plot_summary('shap_summary.png')
                    if len(X) > 0:
                        self.shap_explainer.plot_waterfall(0, 'shap_waterfall_example.png')
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        return explanations
    
    def save_models(self, model_dir: str = "models") -> None:
        """
        Save fitted models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        if not self.models_fitted:
            raise ValueError("Models must be fitted before saving")
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save models
        joblib.dump(self.isolation_forest.model, model_path / "isolation_forest.pkl")
        joblib.dump(self.feature_engineer.scaler, model_path / "scaler.pkl")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(metadata, model_path / "metadata.pkl")
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "models") -> None:
        """
        Load models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        # Load models
        self.isolation_forest.model = joblib.load(model_path / "isolation_forest.pkl")
        self.feature_engineer.scaler = joblib.load(model_path / "scaler.pkl")
        
        # Load metadata
        metadata = joblib.load(model_path / "metadata.pkl")
        self.feature_names = metadata['feature_names']
        self.feature_engineer.feature_columns = self.feature_names
        
        self.models_fitted = True
        logger.info(f"Models loaded from {model_dir}")


def main():
    """Example usage of ML anomaly detection."""
    # Create sample data
    np.random.seed(42)
    n_precincts = 1000
    
    # Create realistic election data with some anomalies
    sample_data = pd.DataFrame({
        'County': np.random.choice(['County A', 'County B', 'County C'], n_precincts),
        'Precinct': [f'Precinct {i}' for i in range(n_precincts)],
        'Lat': np.random.uniform(40, 42, n_precincts),
        'Lon': np.random.uniform(-75, -73, n_precincts),
        'Registered_Dem': np.random.randint(200, 2000, n_precincts),
        'Registered_Rep': np.random.randint(200, 2000, n_precincts),
        'Turnout_Percent': np.random.beta(5, 3) * 100,
    })
    
    # Add realistic vote patterns
    sample_data['Total_Votes'] = (
        (sample_data['Registered_Dem'] + sample_data['Registered_Rep']) * 
        sample_data['Turnout_Percent'] / 100
    ).astype(int)
    
    # Base Harris share with some correlation to Dem registration
    dem_ratio = sample_data['Registered_Dem'] / (sample_data['Registered_Dem'] + sample_data['Registered_Rep'])
    sample_data['Harris_Share'] = np.clip(
        0.3 + 0.4 * dem_ratio + np.random.normal(0, 0.1, n_precincts),
        0, 1
    )
    
    sample_data['Votes_Harris'] = (sample_data['Total_Votes'] * sample_data['Harris_Share']).astype(int)
    sample_data['Votes_Trump'] = sample_data['Total_Votes'] - sample_data['Votes_Harris']
    
    # Add some artificial anomalies
    anomaly_indices = np.random.choice(n_precincts, 50, replace=False)
    sample_data.loc[anomaly_indices, 'Turnout_Percent'] *= 1.3  # Boost turnout
    sample_data.loc[anomaly_indices, 'Harris_Share'] *= 1.2    # Boost vote share
    
    # Run ML analysis
    detector = MLAnomalyDetector()
    detector.fit_models(sample_data)
    results = detector.predict_anomalies(sample_data)
    
    # Generate explanations
    explanations = detector.explain_predictions(sample_data)
    
    print(f"ML analysis completed on {len(results)} precincts")
    print(f"Isolation Forest anomalies: {results['IF_Anomaly_Flag'].sum()}")
    print(f"DBSCAN anomalies: {results['DBSCAN_Anomaly_Flag'].sum()}")
    print(f"ML composite anomalies: {results['ML_Anomaly_Flag'].sum()}")
    
    if 'isolation_forest_importance' in explanations:
        print("\nTop 5 most important features:")
        top_features = explanations['isolation_forest_importance'].head()
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
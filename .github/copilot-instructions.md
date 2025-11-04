# Election Truth Alliance Data Dashboard - AI Coding Agent Instructions

## Project Overview
A comprehensive Python system for statistical anomaly detection in precinct-level election data. Implements advanced statistical methods (Shpilkin analysis, Moran's I spatial autocorrelation), machine learning algorithms (Isolation Forest, DBSCAN), and interactive Streamlit dashboard for election integrity analysis.

## Architecture & Key Components

### Core Processing Pipeline
- **`data_ingestion.py`**: ElectionDataIngester class handles CSV loading, validation, normalization, and feature engineering
- **`statistical_models.py`**: StatisticalAnomalyDetector implements Shpilkin turnout analysis, spatial autocorrelation, entropy analysis
- **`ml_models.py`**: MLAnomalyDetector provides Isolation Forest, DBSCAN clustering, SHAP interpretability
- **`visualization.py`**: ComprehensiveVisualizer creates Plotly/Folium maps, statistical plots, interactive visualizations
- **`dashboard.py`**: Main Streamlit application with tabbed interface for data exploration and analysis
- **`llm_integration.py`**: Optional LLM reasoning agent (GPT-4, Claude, Command-R+) for intelligent analysis

### Data Flow Architecture
1. **Ingestion**: CSV → validation → normalization → derived features → processed DataFrame
2. **Analysis**: Statistical methods + ML algorithms → anomaly scores + flags → composite scoring
3. **Visualization**: Interactive plots + geospatial maps + statistical distributions
4. **Dashboard**: Multi-tab Streamlit UI → filtering → export capabilities → LLM summaries

## Development Workflows

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run main dashboard
streamlit run dashboard.py

# Test individual modules
python data_ingestion.py    # Sample data processing
python statistical_models.py # Shpilkin analysis demo
python ml_models.py         # ML anomaly detection
python visualization.py     # Plot generation
```

### Testing Analysis Pipeline
```bash
# Generate sample data and run full analysis
python -c "
from data_ingestion import ElectionDataIngester
from statistical_models import StatisticalAnomalyDetector
from ml_models import MLAnomalyDetector

# Process sample data
ingester = ElectionDataIngester()
# ... (see module main() functions for examples)
"
```

### Configuration Management
```bash
# Edit analysis parameters
vim config.yaml

# Key sections:
# - data.validation: lat/lon ranges, turnout limits
# - statistics.shpilkin: polynomial degree, confidence intervals  
# - ml.isolation_forest: contamination rate, n_estimators
# - llm.providers: API keys and model selection
```

## Project Conventions

### Class Architecture Patterns
- **Detector classes** (StatisticalAnomalyDetector, MLAnomalyDetector): fit() → predict() → explain() workflow
- **Analyzer classes** (ShpilkinAnalyzer, SpatialAnalyzer): prepare_data() → fit_model() → detect_anomalies()
- **Visualizer classes**: create_*_plot() methods return Plotly Figure objects
- **Config-driven**: All classes accept config dict in constructor, fall back to reasonable defaults

### Data Processing Standards
- **Required columns**: County, Precinct, Lat, Lon, Registered_Dem, Registered_Rep, Votes_Harris, Votes_Trump, Total_Votes, Turnout_Percent
- **Derived features**: Always add Harris_Share, Trump_Share, Total_Registered, Turnout_Discrepancy, Precinct_ID
- **Validation**: Lat [-90,90], Lon [-180,180], Turnout [0,100], non-negative vote counts
- **Missing values**: County-level interpolation → median imputation → drop if excessive

### Anomaly Detection Methodology
- **Shpilkin analysis**: Polynomial regression on turnout vs vote share, residual analysis, 2σ threshold
- **Spatial clustering**: Moran's I (global/local), Queen/Rook/KNN weights, p < 0.05 significance
- **ML anomaly detection**: Isolation Forest (contamination=0.1) + DBSCAN (eps=0.5, min_samples=5)
- **Composite scoring**: Weighted combination (Shpilkin 0.4, ML 0.4, Spatial 0.2)

## Key Files & Implementation Details

### `data_ingestion.py` - ElectionDataIngester
- **Key methods**: `process_file()` (main pipeline), `validate_ranges()`, `handle_missing_values()`, `add_derived_features()`
- **Validation logic**: Coordinate bounds checking, turnout percentage limits, non-negative vote validation
- **Feature engineering**: Registration ratios, vote shares, county-normalized deviations, competitive margins

### `statistical_models.py` - Core Analysis
- **ShpilkinAnalyzer**: `create_turnout_bins()` → `fit_expected_curve()` → `detect_anomalies()` → `analyze_excess_votes()`
- **SpatialAnalyzer**: Requires libpysal/esda, `create_spatial_weights()` → `calculate_global_moran()` → `calculate_local_moran()`
- **EntropyAnalyzer**: Last digit distribution analysis, round number preference detection, Benford's Law testing

### `ml_models.py` - Machine Learning Pipeline  
- **FeatureEngineer**: Creates 20+ features (vote margins, registration efficiency, turnout categories, county deviations)
- **IsolationForestDetector**: Anomaly scoring + permutation feature importance + SHAP integration
- **DBSCANDetector**: Clustering + noise detection, silhouette score validation, cluster characteristic analysis

### `visualization.py` - Interactive Plotting
- **ShpilkinPlotter**: `create_turnout_scatter()` with polynomial overlay, residual analysis plots
- **GeospatialVisualizer**: Folium maps, Plotly choropleth, density heatmaps, county aggregation
- **StatisticalPlotter**: Distribution analysis, correlation heatmaps, anomaly summary bar charts

### `dashboard.py` - Streamlit Interface
- **Multi-tab structure**: Data Explorer → Statistical Analysis → ML Detection → Geospatial → Reports
- **Session state management**: Persistent data loading, analysis results, filter states
- **Dynamic filtering**: County selection, turnout ranges, vote count minimums
- **Export capabilities**: CSV downloads for raw data, analysis results, anomalies-only

## External Dependencies & Integration Points

### Required Libraries
- **Analysis**: pandas, numpy, scipy, scikit-learn, statsmodels
- **Geospatial**: geopandas, folium, libpysal, esda (spatial autocorrelation)
- **Visualization**: plotly, matplotlib, seaborn  
- **Dashboard**: streamlit (main UI framework)
- **ML Interpretability**: shap (model explanations)

### Optional LLM Integration
- **Providers**: openai (GPT-4), anthropic (Claude), cohere (Command-R+)
- **Usage pattern**: `agent.initialize_provider(provider, api_key)` → `agent.analyze_anomalies(df)` → `agent.generate_executive_summary()`
- **Prompt engineering**: Structured analysis of statistical evidence, geographic context, audit prioritization

### Configuration Dependencies
- **YAML config**: All analysis parameters externalized to `config.yaml`
- **Environment variables**: LLM API keys, optional database connections
- **Graceful degradation**: Missing optional dependencies disable features without breaking core functionality

## Troubleshooting & Common Issues

### Data Loading Problems
- **"Missing required columns"**: Check CSV headers match exact column names (case-sensitive)
- **Unicode errors**: ElectionDataIngester tries utf-8, latin1, cp1252 encodings automatically
- **Invalid coordinates**: Validation flags out-of-range lat/lon, sets to NaN for interpolation

### Analysis Failures  
- **"Insufficient data for curve fitting"**: Need ≥5 precincts per turnout bin, adjust `bin_size` in config
- **Spatial analysis errors**: Install libpysal/esda packages, check coordinate validity
- **SHAP memory issues**: Reduce `sample_size` in config, use `max_display` parameter

### Dashboard Issues
- **Streamlit caching**: Clear with `st.cache_data.clear()` or restart application
- **Large dataset performance**: Use data filtering, consider sampling for visualization
- **Missing visualizations**: Check browser JavaScript console, verify Plotly/Folium dependencies

### Development Workflow
- **Module testing**: Each .py file has `main()` function with sample data and usage examples
- **Configuration changes**: Restart Streamlit dashboard after editing `config.yaml`  
- **Adding new analysis methods**: Extend detector classes, update dashboard tabs, add visualization methods

---

**Key Insight**: This codebase emphasizes modular architecture with clear separation between data processing, analysis algorithms, visualization, and user interface. Each major component can be tested independently, and the configuration-driven approach allows tuning analysis parameters without code changes.
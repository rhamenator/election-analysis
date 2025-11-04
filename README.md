# Election Truth Alliance Data Dashboard

A comprehensive Python system for statistical anomaly detection in precinct-level election data, implementing advanced statistical methods, machine learning algorithms, and interactive visualizations to identify potential irregularities requiring audit attention.

## 🎯 Features

### Statistical Analysis
- **Shpilkin-style turnout analysis** - Polynomial regression on turnout vs vote share with residual analysis
- **Spatial autocorrelation** - Moran's I global and local statistics for geographic clustering
- **Entropy analysis** - Klimek-style digit pattern detection and round number preferences
- **Excess vote detection** - Statistical analysis of high-turnout anomalies

### Machine Learning
- **Isolation Forest** - Unsupervised anomaly detection with feature importance analysis
- **DBSCAN clustering** - Density-based spatial clustering for outlier identification  
- **SHAP interpretability** - Model explainability and feature contribution analysis
- **Composite scoring** - Weighted combination of multiple detection methods

### Geospatial Visualization
- **Interactive maps** - Folium-based choropleth and heatmap visualizations
- **County-level aggregation** - Geographic patterns and boundary analysis
- **Anomaly overlays** - Color-coded severity indicators on geographic data

### Dashboard Interface
- **Streamlit web app** - Interactive data exploration and analysis interface
- **Real-time filtering** - Dynamic county, turnout, and vote count filters
- **Export capabilities** - CSV downloads for raw data, results, and anomalies
- **Multi-tab navigation** - Organized workflow from data upload to final reports

### Optional LLM Integration
- **AI-powered analysis** - Natural language explanations of statistical findings
- **Audit prioritization** - Intelligent recommendations for investigation priorities
- **Executive summaries** - Automated report generation with actionable insights

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd election-analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard:**
```bash
streamlit run dashboard.py
```

4. **Access the application:**
Open your browser to `http://localhost:8501`

### Basic Usage

1. **Upload Data**: Use the sidebar to upload a CSV file with the required columns
2. **Configure Analysis**: Select analysis methods and candidate for examination
3. **Run Analysis**: Click "Run Analysis" to execute statistical and ML detection
4. **Explore Results**: Navigate through tabs to examine findings and visualizations
5. **Export Reports**: Download results, anomaly lists, and summary statistics

## 📊 Data Format

Your CSV file must contain these columns:

| Column | Description | Type |
|--------|-------------|------|
| `County` | County name | String |
| `Precinct` | Precinct identifier | String |
| `Lat` | Latitude coordinate | Float |
| `Lon` | Longitude coordinate | Float |
| `Registered_Dem` | Democratic registered voters | Integer |
| `Registered_Rep` | Republican registered voters | Integer |
| `Votes_Harris` | Votes for Harris | Integer |
| `Votes_Trump` | Votes for Trump | Integer |
| `Total_Votes` | Total votes cast | Integer |
| `Turnout_Percent` | Voter turnout percentage | Float |

**Example:**
```csv
County,Precinct,Lat,Lon,Registered_Dem,Registered_Rep,Votes_Harris,Votes_Trump,Total_Votes,Turnout_Percent
Adams County,Precinct 1,40.1234,-75.5678,1200,1100,580,520,1150,52.3
Baker County,Precinct 2,40.2345,-75.6789,800,1300,380,720,1120,53.3
```

## 🔬 Analysis Methods

### Shpilkin Analysis
Identifies statistical anomalies by analyzing the relationship between voter turnout and vote share:
- Fits polynomial regression curves to expected patterns
- Calculates residuals and confidence intervals  
- Flags precincts with significant deviations from expected behavior
- Detects "excess votes" at high turnout levels

### Spatial Autocorrelation
Uses Moran's I statistics to detect geographic clustering:
- **Global Moran's I**: Overall spatial autocorrelation in vote patterns
- **Local Moran's I (LISA)**: Hotspot and coldspot identification
- **Cluster classification**: High-High, Low-Low, High-Low, Low-High patterns

### Machine Learning Anomaly Detection
Applies unsupervised learning for pattern recognition:
- **Isolation Forest**: Identifies outliers based on feature isolation difficulty
- **DBSCAN**: Finds density-based clusters and noise points
- **Feature engineering**: Creates 20+ electoral and geographic features
- **SHAP analysis**: Explains model predictions and feature importance

### Statistical Entropy Analysis
Examines digit patterns for artificial manipulation:
- **Last digit analysis**: Tests for uniform distribution in vote counts
- **Round number detection**: Identifies preference for multiples of 5, 10, 25, 50, 100
- **Benford's Law testing**: First digit distribution analysis for large vote counts

## 🗺️ Geospatial Features

### Interactive Maps
- **Folium integration**: Pan, zoom, and click functionality
- **Color-coded markers**: Anomaly severity visualization
- **Popup information**: Detailed precinct statistics on hover
- **Layer controls**: Toggle between different metrics and visualizations

### Geographic Analysis
- **County aggregation**: Statistical summaries by administrative boundaries  
- **Distance calculations**: Proximity-based feature engineering
- **Spatial weights**: Queen, Rook, and K-nearest neighbor matrices
- **Heatmap generation**: Density visualization of anomaly concentrations

## ⚙️ Configuration

### Configuration File (`config.yaml`)
```yaml
# Data validation settings
data:
  validation:
    lat_range: [-90, 90]
    lon_range: [-180, 180]
    turnout_range: [0, 100]

# Statistical analysis parameters
statistics:
  shpilkin:
    polynomial_degree: 3
    confidence_interval: 0.95
  spatial:
    moran_i_weights: 'queen'

# Machine learning settings  
ml:
  isolation_forest:
    contamination: 0.1
    n_estimators: 100
  dbscan:
    eps: 0.5
    min_samples: 5

# LLM integration (optional)
llm:
  enabled: false
  providers:
    openai:
      model: 'gpt-4'
```

## 🤖 LLM Integration Setup

Enable AI-powered analysis and reporting:

1. **Install LLM packages:**
```bash
pip install openai anthropic cohere
```

2. **Configure API access:**
```python
from llm_integration import AnomalyReasoningAgent

agent = AnomalyReasoningAgent()
agent.initialize_provider('openai', 'your-api-key')

# Generate intelligent analysis
analyses = agent.analyze_anomalies(results_df)
summary = agent.generate_executive_summary(analyses)
```

3. **Supported providers:**
- **OpenAI GPT-4** - Most comprehensive analysis capabilities
- **Anthropic Claude** - Strong reasoning and explanation abilities  
- **Cohere Command-R+** - Efficient analysis with good accuracy

## 📋 Dashboard Tabs

### 1. Data Explorer
- Dataset overview and quality metrics
- Interactive data table with search and filtering
- Column selection and raw data export
- Missing value and validation issue reporting

### 2. Statistical Analysis  
- Shpilkin scatter plots with polynomial fitting
- Residual analysis and confidence intervals
- Statistical distribution visualizations
- Top anomaly rankings and explanations

### 3. ML Anomaly Detection
- Isolation Forest and DBSCAN results  
- Feature importance analysis and SHAP plots
- Composite anomaly scoring methodology
- Model performance and cluster analysis

### 4. Geospatial Analysis
- Interactive choropleth and heatmap visualizations
- County-level statistical aggregation
- Spatial autocorrelation results and cluster maps
- Geographic filtering and region-specific analysis

### 5. Reports & Export
- Executive summary with key findings
- Downloadable CSV files (raw data, results, anomalies only)
- Statistical summary tables and audit recommendations
- Comprehensive analysis documentation

## 🔧 Development

### Project Structure
```
election-analysis/
├── config.yaml              # Configuration settings
├── requirements.txt          # Python dependencies  
├── data_ingestion.py        # CSV loading and validation
├── statistical_models.py    # Shpilkin, Moran's I, entropy analysis
├── ml_models.py            # Isolation Forest, DBSCAN, SHAP
├── visualization.py        # Plotly, Folium, statistical plots
├── dashboard.py           # Streamlit web interface
├── llm_integration.py     # Optional AI analysis (GPT, Claude, etc.)
└── .github/
    └── copilot-instructions.md  # AI coding agent guidelines
```

### Running Tests
```bash
# Run statistical analysis on sample data
python statistical_models.py

# Test ML anomaly detection  
python ml_models.py

# Generate sample visualizations
python visualization.py

# Test LLM integration
python llm_integration.py
```

### Adding New Analysis Methods

1. **Statistical methods**: Extend `StatisticalAnomalyDetector` class
2. **ML algorithms**: Add new detectors to `MLAnomalyDetector`  
3. **Visualizations**: Create new plotters in `ComprehensiveVisualizer`
4. **Dashboard tabs**: Modify `DashboardApp.render_main_content()`

## 📚 References

### Academic Literature
- **Shpilkin, S.** (2018). Statistical analysis of Russian election data. *Statistical Methods and Applications*
- **Klimek, P.** (2012). Forensic analysis of Austrian presidential election data. *Statistical Mechanics and its Applications*  
- **Moran, P.A.P.** (1950). Notes on continuous stochastic phenomena. *Biometrika*

### Technical Documentation
- **Isolation Forest**: Liu, F.T. et al. (2008). *Isolation Forest*, IEEE ICDM
- **DBSCAN**: Ester, M. et al. (1996). *A density-based algorithm*, KDD-96
- **SHAP**: Lundberg, S.M. (2017). *A unified approach to interpreting model predictions*, NIPS

## 📄 License

This project is developed for election integrity analysis and transparency. Please ensure compliance with local election laws and data privacy regulations when using with actual election data.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/analysis-method`)
3. Commit changes (`git commit -am 'Add new analysis method'`)
4. Push to branch (`git push origin feature/analysis-method`) 
5. Create a Pull Request

For questions or support, please open an issue in the repository.

---

**⚠️ Important Notice**: This software is provided for research and educational purposes. When analyzing actual election data, ensure compliance with applicable laws, regulations, and data privacy requirements. Always verify findings through appropriate official channels before taking any actions based on anomaly detection results.
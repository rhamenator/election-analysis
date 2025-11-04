"""
Election Truth Alliance Data Dashboard

Streamlit application for comprehensive election anomaly detection and analysis.
Provides interactive interface for data upload, statistical analysis, ML anomaly detection,
geospatial visualization, and downloadable reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
import zipfile
import logging
from typing import Dict, List, Optional, Any
import warnings

# Import our modules
from data_ingestion import ElectionDataIngester
from statistical_models import StatisticalAnomalyDetector
from ml_models import MLAnomalyDetector
from visualization import ComprehensiveVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Election Truth Alliance Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-high { color: #ff4b4b; font-weight: bold; }
    .anomaly-medium { color: #ffa500; font-weight: bold; }
    .anomaly-low { color: #90ee90; }
    .normal { color: #1e88e5; }
</style>
""", unsafe_allow_html=True)


class DashboardApp:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize dashboard application."""
        self.data_ingester = ElectionDataIngester()
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        self.visualizer = ComprehensiveVisualizer()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def run(self):
        """Run the dashboard application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📊 Election Truth Alliance Data Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Comprehensive Statistical Anomaly Detection for Precinct-Level Election Data**
        
        This dashboard implements advanced statistical and machine learning methods to detect 
        anomalies in election data, including Shpilkin-style turnout analysis, spatial 
        autocorrelation, and unsupervised ML techniques.
        """)
        
        st.divider()
    
    def render_sidebar(self):
        """Render the sidebar with controls and filters."""
        st.sidebar.title("📋 Controls & Filters")
        
        # Data Upload Section
        st.sidebar.header("1. Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload election data CSV with required columns"
        )
        
        if uploaded_file is not None and not st.session_state.data_loaded:
            self.load_data(uploaded_file)
        
        # Analysis Controls
        if st.session_state.data_loaded:
            st.sidebar.header("2. Analysis Settings")
            
            # Candidate selection
            vote_share_columns = [col for col in st.session_state.processed_data.columns 
                                if 'Share' in col or 'Percent' in col]
            
            if vote_share_columns:
                candidate_column = st.sidebar.selectbox(
                    "Select candidate for analysis",
                    vote_share_columns,
                    index=0 if 'Harris_Share' in vote_share_columns else 0
                )
                st.session_state.candidate_column = candidate_column
            
            # Analysis type selection
            analysis_types = st.sidebar.multiselect(
                "Select analysis methods",
                ["Statistical Analysis", "ML Anomaly Detection", "Spatial Analysis"],
                default=["Statistical Analysis", "ML Anomaly Detection"]
            )
            
            # Run analysis button
            if st.sidebar.button("🔍 Run Analysis", type="primary"):
                self.run_analysis(analysis_types)
        
        # Data Filters (if data loaded)
        if st.session_state.data_loaded and st.session_state.processed_data is not None:
            st.sidebar.header("3. Data Filters")
            self.render_data_filters()
    
    def render_data_filters(self):
        """Render data filtering controls."""
        data = st.session_state.processed_data
        
        # County filter
        counties = sorted(data['County'].unique())
        selected_counties = st.sidebar.multiselect(
            "Filter by County",
            counties,
            default=counties
        )
        
        # Turnout filter
        turnout_range = st.sidebar.slider(
            "Turnout Range (%)",
            min_value=float(data['Turnout_Percent'].min()),
            max_value=float(data['Turnout_Percent'].max()),
            value=(float(data['Turnout_Percent'].min()), float(data['Turnout_Percent'].max())),
            step=1.0
        )
        
        # Vote count filter
        if 'Total_Votes' in data.columns:
            min_votes = st.sidebar.number_input(
                "Minimum Total Votes",
                min_value=0,
                max_value=int(data['Total_Votes'].max()),
                value=0
            )
        else:
            min_votes = 0
        
        # Apply filters
        filtered_data = data[
            (data['County'].isin(selected_counties)) &
            (data['Turnout_Percent'] >= turnout_range[0]) &
            (data['Turnout_Percent'] <= turnout_range[1])
        ]
        
        if 'Total_Votes' in data.columns:
            filtered_data = filtered_data[filtered_data['Total_Votes'] >= min_votes]
        
        st.session_state.filtered_data = filtered_data
        
        # Show filter summary
        st.sidebar.info(f"""
        **Filtered Data Summary**
        - Precincts: {len(filtered_data):,}
        - Counties: {filtered_data['County'].nunique()}
        - Total Votes: {filtered_data.get('Total_Votes', pd.Series([0])).sum():,}
        """)
    
    def load_data(self, uploaded_file):
        """Load and process uploaded data."""
        try:
            with st.spinner("Loading and validating data..."):
                # Save uploaded file temporarily
                temp_path = Path("temp_upload.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the data
                processed_data, summary = self.data_ingester.process_file(temp_path)
                
                # Clean up temp file
                temp_path.unlink()
                
                # Store in session state
                st.session_state.raw_data = uploaded_file
                st.session_state.processed_data = processed_data
                st.session_state.filtered_data = processed_data
                st.session_state.data_summary = summary
                st.session_state.data_loaded = True
                
                st.sidebar.success(f"✅ Loaded {len(processed_data):,} precincts successfully!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    def run_analysis(self, analysis_types):
        """Run selected analysis methods."""
        if not st.session_state.data_loaded:
            st.error("Please upload data first!")
            return
        
        try:
            with st.spinner("Running analysis..."):
                data = st.session_state.processed_data.copy()
                candidate_column = getattr(st.session_state, 'candidate_column', 'Harris_Share')
                
                # Statistical Analysis
                if "Statistical Analysis" in analysis_types:
                    data = self.statistical_detector.run_full_analysis(data, candidate_column)
                
                # ML Anomaly Detection
                if "ML Anomaly Detection" in analysis_types:
                    self.ml_detector.fit_models(data)
                    data = self.ml_detector.predict_anomalies(data)
                
                # Store results
                st.session_state.analysis_results = data
                st.session_state.analysis_complete = True
                
                st.sidebar.success("✅ Analysis complete!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Analysis failed: {str(e)}")
    
    def render_main_content(self):
        """Render the main content area with tabs."""
        if not st.session_state.data_loaded:
            self.render_welcome_screen()
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Explorer", 
            "📈 Statistical Analysis", 
            "🤖 ML Anomaly Detection",
            "🗺️ Geospatial Analysis",
            "📋 Reports & Export"
        ])
        
        with tab1:
            self.render_data_explorer()
        
        with tab2:
            self.render_statistical_analysis()
        
        with tab3:
            self.render_ml_analysis()
        
        with tab4:
            self.render_geospatial_analysis()
        
        with tab5:
            self.render_reports()
    
    def render_welcome_screen(self):
        """Render welcome screen when no data is loaded."""
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Upload your election data** using the sidebar file uploader
        2. **Configure analysis settings** in the sidebar
        3. **Run analysis** to detect statistical anomalies
        4. **Explore results** using the interactive tabs
        
        ### 📋 Required Data Format
        Your CSV file should contain these columns:
        - `County` - County name
        - `Precinct` - Precinct identifier
        - `Lat`, `Lon` - Geographic coordinates
        - `Registered_Dem`, `Registered_Rep` - Registered voters by party
        - `Votes_Harris`, `Votes_Trump` - Vote counts by candidate
        - `Total_Votes` - Total votes cast
        - `Turnout_Percent` - Voter turnout percentage
        
        ### 🔬 Analysis Methods
        - **Shpilkin Analysis**: Turnout vs vote share patterns
        - **Spatial Autocorrelation**: Geographic clustering analysis
        - **ML Anomaly Detection**: Isolation Forest and DBSCAN
        - **Statistical Entropy**: Digit pattern analysis
        """)
        
        # Sample data download
        if st.button("📥 Download Sample Data Format"):
            sample_data = self.create_sample_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample_election_data.csv",
                data=csv,
                file_name="sample_election_data.csv",
                mime="text/csv"
            )
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for download."""
        np.random.seed(42)
        n_precincts = 100
        
        sample_data = pd.DataFrame({
            'County': np.random.choice(['Adams County', 'Baker County', 'Clark County'], n_precincts),
            'Precinct': [f'Precinct {i+1}' for i in range(n_precincts)],
            'Lat': np.random.uniform(39.0, 41.0, n_precincts),
            'Lon': np.random.uniform(-85.0, -80.0, n_precincts),
            'Registered_Dem': np.random.randint(200, 1500, n_precincts),
            'Registered_Rep': np.random.randint(200, 1500, n_precincts),
            'Turnout_Percent': np.random.uniform(35, 85, n_precincts),
        })
        
        # Calculate derived fields
        total_registered = sample_data['Registered_Dem'] + sample_data['Registered_Rep']
        sample_data['Total_Votes'] = (total_registered * sample_data['Turnout_Percent'] / 100).astype(int)
        
        # Realistic vote distribution
        dem_base = sample_data['Registered_Dem'] / total_registered
        sample_data['Harris_Share'] = np.clip(
            dem_base * 0.8 + np.random.normal(0, 0.1, n_precincts), 0, 1
        )
        
        sample_data['Votes_Harris'] = (sample_data['Total_Votes'] * sample_data['Harris_Share']).astype(int)
        sample_data['Votes_Trump'] = sample_data['Total_Votes'] - sample_data['Votes_Harris']
        
        return sample_data
    
    def render_data_explorer(self):
        """Render data exploration tab."""
        st.header("📊 Data Explorer")
        
        data = st.session_state.filtered_data
        
        # Data summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Precincts", f"{len(data):,}")
        with col2:
            st.metric("Counties", data['County'].nunique())
        with col3:
            if 'Total_Votes' in data.columns:
                st.metric("Total Votes", f"{data['Total_Votes'].sum():,}")
            else:
                st.metric("Total Votes", "N/A")
        with col4:
            if 'Turnout_Percent' in data.columns:
                st.metric("Avg Turnout", f"{data['Turnout_Percent'].mean():.1f}%")
            else:
                st.metric("Avg Turnout", "N/A")
        
        # Data quality indicators
        st.subheader("📋 Data Quality Summary")
        
        if 'data_summary' in st.session_state:
            summary = st.session_state.data_summary
            quality = summary.get('data_quality', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                missing_coords = quality.get('missing_coordinates', 0)
                st.metric("Missing Coordinates", missing_coords, 
                         delta=f"{missing_coords/len(data)*100:.1f}%" if len(data) > 0 else "0%")
            
            with col2:
                zero_votes = quality.get('zero_votes', 0)
                st.metric("Zero Votes", zero_votes,
                         delta=f"{zero_votes/len(data)*100:.1f}%" if len(data) > 0 else "0%")
            
            with col3:
                high_turnout = quality.get('high_turnout', 0)
                st.metric("High Turnout (>100%)", high_turnout,
                         delta=f"{high_turnout/len(data)*100:.1f}%" if len(data) > 0 else "0%")
            
            with col4:
                duplicates = quality.get('duplicate_precincts', 0)
                st.metric("Duplicate Precincts", duplicates)
        
        # Raw data table
        st.subheader("🔍 Raw Data Preview")
        
        # Column selector
        all_columns = data.columns.tolist()
        default_columns = ['County', 'Precinct', 'Turnout_Percent', 'Total_Votes', 
                          'Votes_Harris', 'Votes_Trump']
        display_columns = [col for col in default_columns if col in all_columns]
        
        selected_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=display_columns
        )
        
        if selected_columns:
            # Add search functionality
            search_term = st.text_input("🔍 Search precincts (by county or precinct name)")
            
            display_data = data[selected_columns].copy()
            
            if search_term:
                mask = (
                    display_data['County'].str.contains(search_term, case=False, na=False) |
                    display_data['Precinct'].str.contains(search_term, case=False, na=False)
                )
                display_data = display_data[mask]
            
            st.dataframe(display_data, use_container_width=True, height=400)
            
            # Download filtered data
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_election_data.csv",
                mime="text/csv"
            )
    
    def render_statistical_analysis(self):
        """Render statistical analysis tab."""
        st.header("📈 Statistical Analysis")
        
        if not st.session_state.analysis_complete:
            st.info("Please run statistical analysis first using the sidebar controls.")
            return
        
        data = st.session_state.analysis_results
        candidate_column = getattr(st.session_state, 'candidate_column', 'Harris_Share')
        
        # Analysis summary
        st.subheader("📊 Analysis Summary")
        
        anomaly_columns = [col for col in data.columns if 'Anomaly_Flag' in col]
        
        if anomaly_columns:
            cols = st.columns(len(anomaly_columns))
            
            for i, col in enumerate(anomaly_columns):
                anomaly_count = data[col].sum()
                anomaly_pct = anomaly_count / len(data) * 100
                
                with cols[i]:
                    method_name = col.replace('_Anomaly_Flag', '').replace('_', ' ')
                    st.metric(
                        f"{method_name} Anomalies",
                        f"{anomaly_count}",
                        delta=f"{anomaly_pct:.1f}% of precincts"
                    )
        
        # Shpilkin Analysis
        if all(col in data.columns for col in ['Turnout_Percent', candidate_column]):
            st.subheader("🎯 Shpilkin Turnout Analysis")
            
            # Create scatter plot
            fig = self.visualizer.shpilkin.create_turnout_scatter(
                data, candidate_column, 'Shpilkin_Anomaly_Flag' if 'Shpilkin_Anomaly_Flag' in data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual analysis
            if 'Vote_Share_Deviation' in data.columns:
                st.subheader("📊 Residual Analysis")
                residual_fig = self.visualizer.shpilkin.create_residual_plot(data, candidate_column)
                st.plotly_chart(residual_fig, use_container_width=True)
        
        # Statistical distributions
        st.subheader("📊 Statistical Distributions")
        
        score_columns = [col for col in data.columns if 'Score' in col and 'Anomaly' in col][:4]
        
        if score_columns:
            dist_fig = self.visualizer.statistical.create_distribution_plots(data, score_columns)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Top anomalies table
        if 'Shpilkin_Anomaly_Score' in data.columns:
            st.subheader("🚨 Top Statistical Anomalies")
            
            top_anomalies = data.nlargest(10, 'Shpilkin_Anomaly_Score')[
                ['County', 'Precinct', 'Turnout_Percent', candidate_column, 'Shpilkin_Anomaly_Score']
            ]
            st.dataframe(top_anomalies, use_container_width=True)
    
    def render_ml_analysis(self):
        """Render ML analysis tab."""
        st.header("🤖 ML Anomaly Detection")
        
        if not st.session_state.analysis_complete:
            st.info("Please run ML analysis first using the sidebar controls.")
            return
        
        data = st.session_state.analysis_results
        
        # ML Summary
        st.subheader("🎯 ML Detection Summary")
        
        ml_columns = ['IF_Anomaly_Flag', 'DBSCAN_Anomaly_Flag', 'ML_Anomaly_Flag']
        existing_ml_cols = [col for col in ml_columns if col in data.columns]
        
        if existing_ml_cols:
            cols = st.columns(len(existing_ml_cols))
            
            for i, col in enumerate(existing_ml_cols):
                anomaly_count = data[col].sum()
                anomaly_pct = anomaly_count / len(data) * 100
                
                with cols[i]:
                    method_name = col.replace('_Anomaly_Flag', '').replace('_', ' ')
                    st.metric(
                        f"{method_name} Anomalies",
                        f"{anomaly_count}",
                        delta=f"{anomaly_pct:.1f}% of precincts"
                    )
        
        # Feature importance (if available)
        if hasattr(self.ml_detector, 'feature_names') and self.ml_detector.feature_names:
            st.subheader("🔍 Feature Importance")
            
            try:
                # Get feature importance from ML detector
                importance_df = self.ml_detector.isolation_forest.get_feature_importance(
                    self.ml_detector.feature_engineer.prepare_features_for_ml(data),
                    self.ml_detector.feature_names
                )
                
                # Plot feature importance
                fig = go.Figure(data=[
                    go.Bar(
                        x=importance_df['importance'].head(10),
                        y=importance_df['feature'].head(10),
                        orientation='h'
                    )
                ])
                
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate feature importance: {str(e)}")
        
        # ML scores distribution
        ml_score_cols = [col for col in data.columns if col in ['IF_Anomaly_Score', 'ML_Composite_Score']]
        
        if ml_score_cols:
            st.subheader("📊 ML Score Distributions")
            
            fig = self.visualizer.statistical.create_distribution_plots(data, ml_score_cols)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top ML anomalies
        if 'ML_Composite_Score' in data.columns:
            st.subheader("🚨 Top ML Anomalies")
            
            display_cols = ['County', 'Precinct', 'ML_Composite_Score']
            if 'IF_Anomaly_Score' in data.columns:
                display_cols.append('IF_Anomaly_Score')
            if 'DBSCAN_Cluster' in data.columns:
                display_cols.append('DBSCAN_Cluster')
            
            top_ml_anomalies = data.nlargest(10, 'ML_Composite_Score')[display_cols]
            st.dataframe(top_ml_anomalies, use_container_width=True)
    
    def render_geospatial_analysis(self):
        """Render geospatial analysis tab."""
        st.header("🗺️ Geospatial Analysis")
        
        data = st.session_state.filtered_data
        
        if not all(col in data.columns for col in ['Lat', 'Lon']):
            st.error("Geographic coordinates (Lat, Lon) are required for geospatial analysis.")
            return
        
        # Map visualization options
        st.subheader("🎯 Map Visualization Options")
        
        # Select value column for mapping
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        score_columns = [col for col in numeric_columns if 'Score' in col or 'Percent' in col]
        
        map_column = st.selectbox(
            "Select value to visualize",
            score_columns if score_columns else numeric_columns,
            index=0
        )
        
        if map_column:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌍 Interactive Map")
                
                # Create scatter mapbox
                fig = go.Figure(go.Scattermapbox(
                    lat=data['Lat'],
                    lon=data['Lon'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data[map_column],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=map_column.replace('_', ' ').title())
                    ),
                    text=data['County'] + " - " + data['Precinct'],
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        f'{map_column}: %{{marker.color:.3f}}<br>' +
                        '<extra></extra>'
                    )
                ))
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(
                        center=go.layout.mapbox.Center(
                            lat=data['Lat'].mean(),
                            lon=data['Lon'].mean()
                        ),
                        zoom=8
                    ),
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 County-Level Summary")
                
                # County aggregation
                county_stats = data.groupby('County').agg({
                    map_column: ['mean', 'std', 'count'],
                    'Lat': 'mean',
                    'Lon': 'mean'
                }).round(3)
                
                county_stats.columns = ['Mean', 'Std Dev', 'Count', 'Lat', 'Lon']
                county_stats = county_stats.sort_values('Mean', ascending=False)
                
                st.dataframe(county_stats, use_container_width=True)
                
                # County bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=county_stats.index,
                        y=county_stats['Mean'],
                        error_y=dict(type='data', array=county_stats['Std Dev']),
                        marker_color=px.colors.qualitative.Set1
                    )
                ])
                
                fig_bar.update_layout(
                    title=f"County Average: {map_column.replace('_', ' ').title()}",
                    xaxis_title="County",
                    yaxis_title="Average Value",
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Spatial correlation analysis
        if st.session_state.analysis_complete and 'Local_Moran_I' in data.columns:
            st.subheader("🔍 Spatial Autocorrelation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Global Moran's I summary
                if hasattr(data, 'attrs') and 'global_moran' in data.attrs:
                    global_moran = data.attrs['global_moran']
                    st.metric("Global Moran's I", f"{global_moran['I']:.4f}")
                    st.metric("P-value", f"{global_moran['p_value']:.4f}")
                    st.metric("Significant?", "Yes" if global_moran['significant'] else "No")
            
            with col2:
                # Local clusters map
                cluster_fig = go.Figure(go.Scattermapbox(
                    lat=data['Lat'],
                    lon=data['Lon'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data['Local_Moran_I'],
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title="Local Moran's I")
                    ),
                    text=data['Cluster_Type'] if 'Cluster_Type' in data.columns else '',
                    hovertemplate=(
                        'Local Moran I: %{marker.color:.3f}<br>' +
                        'Cluster Type: %{text}<br>' +
                        '<extra></extra>'
                    )
                ))
                
                cluster_fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(
                        center=go.layout.mapbox.Center(
                            lat=data['Lat'].mean(),
                            lon=data['Lon'].mean()
                        ),
                        zoom=8
                    ),
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=400,
                    title="Local Spatial Clusters"
                )
                
                st.plotly_chart(cluster_fig, use_container_width=True)
    
    def render_reports(self):
        """Render reports and export tab."""
        st.header("📋 Reports & Export")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first to generate reports.")
            return
        
        data = st.session_state.filtered_data
        
        # Executive Summary
        st.subheader("📊 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Dataset Overview")
            st.write(f"- **Total Precincts**: {len(data):,}")
            st.write(f"- **Counties**: {data['County'].nunique()}")
            
            if 'Total_Votes' in data.columns:
                st.write(f"- **Total Votes**: {data['Total_Votes'].sum():,}")
            
            if 'Turnout_Percent' in data.columns:
                st.write(f"- **Average Turnout**: {data['Turnout_Percent'].mean():.1f}%")
                st.write(f"- **Turnout Range**: {data['Turnout_Percent'].min():.1f}% - {data['Turnout_Percent'].max():.1f}%")
        
        with col2:
            if st.session_state.analysis_complete:
                st.markdown("### 🚨 Anomaly Detection Results")
                
                anomaly_columns = [col for col in data.columns if 'Anomaly_Flag' in col]
                
                for col in anomaly_columns:
                    method_name = col.replace('_Anomaly_Flag', '').replace('_', ' ')
                    anomaly_count = data[col].sum()
                    anomaly_pct = anomaly_count / len(data) * 100
                    st.write(f"- **{method_name}**: {anomaly_count} ({anomaly_pct:.1f}%)")
        
        # Data Export Options
        st.subheader("📥 Data Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Raw data export
            if st.button("📊 Export Raw Data"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download raw_data.csv",
                    data=csv,
                    file_name="election_raw_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Analysis results export
            if st.session_state.analysis_complete:
                if st.button("🔍 Export Analysis Results"):
                    analysis_data = st.session_state.analysis_results
                    csv = analysis_data.to_csv(index=False)
                    st.download_button(
                        label="Download analysis_results.csv",
                        data=csv,
                        file_name="election_analysis_results.csv",
                        mime="text/csv"
                    )
        
        with col3:
            # Anomalies only export
            if st.session_state.analysis_complete:
                if st.button("🚨 Export Anomalies Only"):
                    analysis_data = st.session_state.analysis_results
                    
                    # Find any anomaly flag column
                    anomaly_columns = [col for col in analysis_data.columns if 'Anomaly_Flag' in col]
                    
                    if anomaly_columns:
                        # Create combined anomaly flag
                        combined_anomaly = analysis_data[anomaly_columns].any(axis=1)
                        anomalies_only = analysis_data[combined_anomaly]
                        
                        csv = anomalies_only.to_csv(index=False)
                        st.download_button(
                            label="Download anomalies.csv",
                            data=csv,
                            file_name="election_anomalies.csv",
                            mime="text/csv"
                        )
        
        # Summary Statistics
        st.subheader("📈 Summary Statistics")
        
        # Select numeric columns for statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        key_columns = [col for col in numeric_columns if any(keyword in col.lower() 
                      for keyword in ['turnout', 'votes', 'share', 'score'])]
        
        if key_columns:
            stats_df = data[key_columns].describe().round(3)
            st.dataframe(stats_df, use_container_width=True)
        
        # Anomaly Details Table
        if st.session_state.analysis_complete:
            st.subheader("🔍 Detailed Anomaly Analysis")
            
            analysis_data = st.session_state.analysis_results
            
            # Create summary of all anomalies
            anomaly_flag_cols = [col for col in analysis_data.columns if 'Anomaly_Flag' in col]
            score_cols = [col for col in analysis_data.columns if 'Score' in col and 'Anomaly' in col]
            
            if anomaly_flag_cols:
                # Combine all anomaly flags
                analysis_data['Any_Anomaly'] = analysis_data[anomaly_flag_cols].any(axis=1)
                
                display_cols = ['County', 'Precinct', 'Turnout_Percent']
                if 'Harris_Share' in analysis_data.columns:
                    display_cols.append('Harris_Share')
                display_cols.extend(anomaly_flag_cols)
                display_cols.extend(score_cols[:3])  # Limit score columns
                
                # Filter to anomalies only
                anomalies_df = analysis_data[analysis_data['Any_Anomaly']][display_cols]
                
                if len(anomalies_df) > 0:
                    st.dataframe(anomalies_df, use_container_width=True, height=400)
                else:
                    st.info("No anomalies detected with current settings.")


def main():
    """Main function to run the Streamlit dashboard."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
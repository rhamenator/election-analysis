"""
Visualization Module

This module creates comprehensive visualizations for election anomaly analysis,
including Shpilkin-style scatter plots, geospatial heatmaps, statistical distributions,
and interactive maps with anomaly overlays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    warnings.warn("Folium not available. Install folium for interactive maps.")
    FOLIUM_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    warnings.warn("GeoPandas not available. County boundaries not supported.")
    GEOPANDAS_AVAILABLE = False

# Configure logging and plotting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ShpilkinPlotter:
    """
    Creates Shpilkin-style turnout vs vote share visualizations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Shpilkin plotter."""
        self.config = config or {
            'figure_size': (12, 8),
            'dpi': 300,
            'colors': {
                'normal': '#1f77b4',
                'anomaly': '#ff7f0e',
                'curve': '#d62728'
            }
        }
    
    def create_turnout_scatter(self, df: pd.DataFrame,
                             candidate_column: str = 'Harris_Share',
                             anomaly_column: Optional[str] = None,
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Shpilkin-style turnout vs vote share scatter plot.
        
        Args:
            df: DataFrame with election data
            candidate_column: Column for candidate vote share
            anomaly_column: Optional column for anomaly flags
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating Shpilkin scatter plot for {candidate_column}")
        
        # Prepare data
        plot_df = df.copy()
        plot_df = plot_df[
            (plot_df['Turnout_Percent'] > 0) & 
            (plot_df['Turnout_Percent'] <= 100) &
            (plot_df[candidate_column] >= 0) &
            (plot_df[candidate_column] <= 1)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points
        if anomaly_column and anomaly_column in df.columns:
            # Separate normal and anomalous points
            normal_mask = plot_df[anomaly_column] == 0
            anomaly_mask = plot_df[anomaly_column] == 1
            
            # Normal points
            fig.add_trace(go.Scatter(
                x=plot_df[normal_mask]['Turnout_Percent'],
                y=plot_df[normal_mask][candidate_column],
                mode='markers',
                name='Normal Precincts',
                marker=dict(
                    size=6,
                    color=self.config['colors']['normal'],
                    opacity=0.6
                ),
                hovertemplate=(
                    'Turnout: %{x:.1f}%<br>' +
                    f'{candidate_column}: %{{y:.1%}}<br>' +
                    'County: %{customdata[0]}<br>' +
                    'Precinct: %{customdata[1]}<extra></extra>'
                ),
                customdata=plot_df[normal_mask][['County', 'Precinct']].values
            ))
            
            # Anomalous points
            if anomaly_mask.any():
                fig.add_trace(go.Scatter(
                    x=plot_df[anomaly_mask]['Turnout_Percent'],
                    y=plot_df[anomaly_mask][candidate_column],
                    mode='markers',
                    name='Anomalous Precincts',
                    marker=dict(
                        size=8,
                        color=self.config['colors']['anomaly'],
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    hovertemplate=(
                        'Turnout: %{x:.1f}%<br>' +
                        f'{candidate_column}: %{{y:.1%}}<br>' +
                        'County: %{customdata[0]}<br>' +
                        'Precinct: %{customdata[1]}<br>' +
                        'ANOMALY<extra></extra>'
                    ),
                    customdata=plot_df[anomaly_mask][['County', 'Precinct']].values
                ))
        else:
            # All points same style
            fig.add_trace(go.Scatter(
                x=plot_df['Turnout_Percent'],
                y=plot_df[candidate_column],
                mode='markers',
                name='Precincts',
                marker=dict(
                    size=6,
                    color=self.config['colors']['normal'],
                    opacity=0.6
                ),
                hovertemplate=(
                    'Turnout: %{x:.1f}%<br>' +
                    f'{candidate_column}: %{{y:.1%}}<br>' +
                    'County: %{customdata[0]}<br>' +
                    'Precinct: %{customdata[1]}<extra></extra>'
                ),
                customdata=plot_df[['County', 'Precinct']].values
            ))
        
        # Add expected curve if available
        if 'Expected_Vote_Share' in df.columns:
            # Sort by turnout for smooth curve
            curve_data = plot_df.sort_values('Turnout_Percent')
            
            fig.add_trace(go.Scatter(
                x=curve_data['Turnout_Percent'],
                y=curve_data['Expected_Vote_Share'],
                mode='lines',
                name='Expected Pattern',
                line=dict(
                    color=self.config['colors']['curve'],
                    width=3
                ),
                hovertemplate='Expected at %{x:.1f}% turnout: %{y:.1%}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Turnout vs {candidate_column.replace("_", " ").title()}',
            xaxis_title='Voter Turnout (%)',
            yaxis_title=f'{candidate_column.replace("_", " ").title()}',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_residual_plot(self, df: pd.DataFrame,
                           candidate_column: str = 'Harris_Share',
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create residual analysis plot.
        
        Args:
            df: DataFrame with analysis results
            candidate_column: Column for candidate vote share
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure object
        """
        if 'Vote_Share_Deviation' not in df.columns:
            raise ValueError("Residual analysis requires 'Vote_Share_Deviation' column")
        
        logger.info("Creating residual analysis plot")
        
        fig = go.Figure()
        
        # Add residual scatter
        fig.add_trace(go.Scatter(
            x=df['Turnout_Percent'],
            y=df['Vote_Share_Deviation'],
            mode='markers',
            name='Residuals',
            marker=dict(
                size=6,
                color=np.abs(df['Vote_Share_Deviation']),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Absolute Deviation")
            ),
            hovertemplate=(
                'Turnout: %{x:.1f}%<br>' +
                'Deviation: %{y:.3f}<br>' +
                'County: %{customdata[0]}<br>' +
                'Precinct: %{customdata[1]}<extra></extra>'
            ),
            customdata=df[['County', 'Precinct']].values
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Expected Value")
        
        # Add confidence bands if available
        if 'Shpilkin_Anomaly_Score' in df.columns:
            threshold = 2.0
            fig.add_hline(y=threshold, line_dash="dot", line_color="orange",
                         annotation_text="Anomaly Threshold (+2σ)")
            fig.add_hline(y=-threshold, line_dash="dot", line_color="orange",
                         annotation_text="Anomaly Threshold (-2σ)")
        
        fig.update_layout(
            title='Residual Analysis: Deviation from Expected Vote Share',
            xaxis_title='Voter Turnout (%)',
            yaxis_title='Vote Share Deviation',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class GeospatialVisualizer:
    """
    Creates geospatial visualizations and interactive maps.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize geospatial visualizer."""
        self.config = config or {
            'default_zoom': 6,
            'tile_layer': 'OpenStreetMap',
            'marker_size': 5,
            'colors': {
                'anomaly_high': '#FF4B4B',
                'anomaly_medium': '#FFA500',
                'anomaly_low': '#90EE90',
                'normal': '#1E88E5'
            }
        }
    
    def create_folium_map(self, df: pd.DataFrame,
                         anomaly_column: str,
                         center_coords: Optional[Tuple[float, float]] = None) -> 'folium.Map':
        """
        Create interactive Folium map with anomaly markers.
        
        Args:
            df: DataFrame with geographic data
            anomaly_column: Column containing anomaly scores
            center_coords: Optional center coordinates (lat, lon)
            
        Returns:
            Folium map object
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError("Folium not available. Install folium for interactive maps.")
        
        logger.info(f"Creating Folium map with {len(df)} precincts")
        
        # Calculate center if not provided
        if center_coords is None:
            center_lat = df['Lat'].mean()
            center_lon = df['Lon'].mean()
        else:
            center_lat, center_lon = center_coords
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config['default_zoom'],
            tiles=self.config['tile_layer']
        )
        
        # Add anomaly markers
        for idx, row in df.iterrows():
            if pd.isna(row['Lat']) or pd.isna(row['Lon']):
                continue
            
            # Determine marker color based on anomaly score
            score = row[anomaly_column]
            if isinstance(score, (int, float)):
                if score > 0.8:
                    color = self.config['colors']['anomaly_high']
                    size = 8
                elif score > 0.5:
                    color = self.config['colors']['anomaly_medium'] 
                    size = 6
                elif score > 0.2:
                    color = self.config['colors']['anomaly_low']
                    size = 5
                else:
                    color = self.config['colors']['normal']
                    size = 4
            else:
                color = self.config['colors']['normal']
                size = 4
            
            # Create popup text
            popup_text = f"""
            <b>{row.get('County', 'Unknown')} - {row.get('Precinct', 'Unknown')}</b><br>
            Turnout: {row.get('Turnout_Percent', 'N/A'):.1f}%<br>
            Vote Share: {row.get('Harris_Share', 'N/A'):.1%}<br>
            Anomaly Score: {score:.3f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=size,
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Anomaly Scores</b></p>
        <p><i class="fa fa-circle" style="color:{high}"></i> High (>0.8)</p>
        <p><i class="fa fa-circle" style="color:{medium}"></i> Medium (0.5-0.8)</p>
        <p><i class="fa fa-circle" style="color:{low}"></i> Low (0.2-0.5)</p>
        <p><i class="fa fa-circle" style="color:{normal}"></i> Normal (<0.2)</p>
        </div>
        '''.format(
            high=self.config['colors']['anomaly_high'],
            medium=self.config['colors']['anomaly_medium'],
            low=self.config['colors']['anomaly_low'],
            normal=self.config['colors']['normal']
        )
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_heatmap(self, df: pd.DataFrame,
                      value_column: str,
                      save_path: Optional[str] = None) -> go.Figure:
        """
        Create density heatmap of anomaly scores.
        
        Args:
            df: DataFrame with geographic data
            value_column: Column to use for heatmap values
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating heatmap for {value_column}")
        
        # Filter valid coordinates
        plot_df = df.dropna(subset=['Lat', 'Lon', value_column])
        
        if len(plot_df) == 0:
            raise ValueError("No valid data for heatmap")
        
        # Create density heatmap
        fig = go.Figure(go.Densitymapbox(
            lat=plot_df['Lat'],
            lon=plot_df['Lon'],
            z=plot_df[value_column],
            radius=10,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=value_column.replace('_', ' ').title())
        ))
        
        # Update layout
        center_lat = plot_df['Lat'].mean()
        center_lon = plot_df['Lon'].mean()
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                zoom=8
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_choropleth_by_county(self, df: pd.DataFrame,
                                  value_column: str,
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Create county-level choropleth map.
        
        Args:
            df: DataFrame with county-level data
            value_column: Column to visualize
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating county choropleth for {value_column}")
        
        # Aggregate by county
        county_stats = df.groupby('County').agg({
            value_column: 'mean',
            'Lat': 'mean',
            'Lon': 'mean',
            'Total_Votes': 'sum'
        }).reset_index()
        
        # Create choropleth using scatter plot (since we don't have county boundaries)
        fig = go.Figure(go.Scattermapbox(
            lat=county_stats['Lat'],
            lon=county_stats['Lon'],
            mode='markers',
            marker=dict(
                size=np.sqrt(county_stats['Total_Votes']) / 10,  # Size by vote count
                color=county_stats[value_column],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=value_column.replace('_', ' ').title()),
                sizemode='diameter'
            ),
            text=county_stats['County'],
            hovertemplate=(
                'County: %{text}<br>' +
                f'{value_column}: %{{marker.color:.3f}}<br>' +
                'Total Votes: %{customdata:,}<extra></extra>'
            ),
            customdata=county_stats['Total_Votes']
        ))
        
        # Update layout
        center_lat = county_stats['Lat'].mean()
        center_lon = county_stats['Lon'].mean()
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                zoom=6
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class StatisticalPlotter:
    """
    Creates statistical distribution and diagnostic plots.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize statistical plotter."""
        self.config = config or {
            'figure_size': (12, 8),
            'color_palette': 'viridis'
        }
    
    def create_distribution_plots(self, df: pd.DataFrame,
                                columns: List[str],
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Create distribution plots for multiple variables.
        
        Args:
            df: DataFrame with data
            columns: List of columns to plot
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure with subplots
        """
        logger.info(f"Creating distribution plots for {len(columns)} variables")
        
        # Calculate subplot dimensions
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        for i, col in enumerate(columns):
            row = (i // n_cols) + 1
            col_idx = (i % n_cols) + 1
            
            if col in df.columns:
                # Create histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        nbinsx=30,
                        name=col,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                    ),
                    row=row, col=col_idx
                )
        
        fig.update_layout(
            title_text="Distribution Analysis",
            showlegend=False,
            width=1000,
            height=300 * n_rows
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame,
                                 anomaly_columns: List[str],
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create correlation heatmap for anomaly detection methods.
        
        Args:
            df: DataFrame with anomaly scores
            anomaly_columns: List of anomaly score columns
            save_path: Optional path to save plot
            
        Returns:
            Plotly heatmap figure
        """
        logger.info("Creating anomaly correlation heatmap")
        
        # Calculate correlation matrix
        corr_data = df[anomaly_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Anomaly Detection Method Correlations',
            width=600,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_anomaly_summary_plot(self, df: pd.DataFrame,
                                  anomaly_columns: List[str],
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Create summary plot of anomaly detection results.
        
        Args:
            df: DataFrame with anomaly results
            anomaly_columns: List of anomaly flag columns
            save_path: Optional path to save plot
            
        Returns:
            Plotly bar chart figure
        """
        logger.info("Creating anomaly summary plot")
        
        # Calculate anomaly counts
        anomaly_counts = {}
        for col in anomaly_columns:
            if col in df.columns:
                anomaly_counts[col] = df[col].sum()
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(anomaly_counts.keys()),
                y=list(anomaly_counts.values()),
                marker_color=px.colors.qualitative.Set1[:len(anomaly_counts)]
            )
        ])
        
        fig.update_layout(
            title='Anomaly Detection Summary',
            xaxis_title='Detection Method',
            yaxis_title='Number of Anomalies',
            width=800,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class ComprehensiveVisualizer:
    """
    Main visualization class that orchestrates all plotting capabilities.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize comprehensive visualizer."""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.config = config.get('visualization', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.config = {}
        
        # Initialize plotters
        self.shpilkin = ShpilkinPlotter(self.config)
        self.geospatial = GeospatialVisualizer(self.config)
        self.statistical = StatisticalPlotter(self.config)
        
    def create_analysis_dashboard(self, df: pd.DataFrame,
                                output_dir: str = "plots") -> Dict[str, Any]:
        """
        Create comprehensive visualization dashboard.
        
        Args:
            df: DataFrame with complete analysis results
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with created plot objects and paths
        """
        logger.info("Creating comprehensive analysis dashboard")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plots = {}
        
        # 1. Shpilkin Analysis Plots
        if all(col in df.columns for col in ['Turnout_Percent', 'Harris_Share']):
            anomaly_col = None
            if 'Shpilkin_Anomaly_Flag' in df.columns:
                anomaly_col = 'Shpilkin_Anomaly_Flag'
            
            shpilkin_scatter = self.shpilkin.create_turnout_scatter(
                df, 'Harris_Share', anomaly_col,
                str(output_path / "shpilkin_scatter.html")
            )
            plots['shpilkin_scatter'] = shpilkin_scatter
            
            if 'Vote_Share_Deviation' in df.columns:
                residual_plot = self.shpilkin.create_residual_plot(
                    df, 'Harris_Share',
                    str(output_path / "residual_analysis.html")
                )
                plots['residual_plot'] = residual_plot
        
        # 2. Geospatial Visualizations
        if all(col in df.columns for col in ['Lat', 'Lon']):
            # Find best anomaly score column
            score_columns = [col for col in df.columns if 'Score' in col and 'Anomaly' in col]
            if score_columns:
                primary_score = score_columns[0]
                
                # Interactive map
                if FOLIUM_AVAILABLE:
                    folium_map = self.geospatial.create_folium_map(df, primary_score)
                    folium_map.save(str(output_path / "interactive_map.html"))
                    plots['interactive_map'] = folium_map
                
                # Heatmap
                heatmap = self.geospatial.create_heatmap(
                    df, primary_score,
                    str(output_path / "anomaly_heatmap.html")
                )
                plots['heatmap'] = heatmap
                
                # County choropleth
                choropleth = self.geospatial.create_choropleth_by_county(
                    df, primary_score,
                    str(output_path / "county_choropleth.html")
                )
                plots['choropleth'] = choropleth
        
        # 3. Statistical Analysis Plots
        # Distribution plots
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        key_columns = [col for col in numeric_columns if any(keyword in col.lower() 
                      for keyword in ['turnout', 'share', 'votes', 'score'])][:6]
        
        if key_columns:
            distributions = self.statistical.create_distribution_plots(
                df, key_columns,
                str(output_path / "distributions.html")
            )
            plots['distributions'] = distributions
        
        # Anomaly correlation heatmap
        anomaly_score_cols = [col for col in df.columns 
                            if 'Score' in col and any(method in col for method in 
                            ['Shpilkin', 'ML', 'Composite', 'IF', 'DBSCAN'])]
        
        if len(anomaly_score_cols) > 1:
            correlation_heatmap = self.statistical.create_correlation_heatmap(
                df, anomaly_score_cols,
                str(output_path / "correlation_heatmap.html")
            )
            plots['correlation_heatmap'] = correlation_heatmap
        
        # Anomaly summary
        anomaly_flag_cols = [col for col in df.columns if col.endswith('_Flag')]
        if anomaly_flag_cols:
            summary_plot = self.statistical.create_anomaly_summary_plot(
                df, anomaly_flag_cols,
                str(output_path / "anomaly_summary.html")
            )
            plots['summary_plot'] = summary_plot
        
        logger.info(f"Created {len(plots)} visualization plots in {output_dir}")
        return plots


def main():
    """Example usage of visualization module."""
    # Create sample data with analysis results
    np.random.seed(42)
    n_precincts = 500
    
    sample_data = pd.DataFrame({
        'County': np.random.choice(['County A', 'County B', 'County C'], n_precincts),
        'Precinct': [f'Precinct {i}' for i in range(n_precincts)],
        'Lat': np.random.uniform(40, 42, n_precincts),
        'Lon': np.random.uniform(-75, -73, n_precincts),
        'Turnout_Percent': np.random.beta(5, 3) * 100,
        'Harris_Share': np.random.beta(2, 2),
        'Total_Votes': np.random.randint(100, 2000, n_precincts),
        
        # Simulated analysis results
        'Shpilkin_Anomaly_Score': np.random.exponential(0.5, n_precincts),
        'ML_Composite_Score': np.random.beta(2, 5, n_precincts),
        'Composite_Anomaly_Score': np.random.beta(1, 4, n_precincts),
        
        # Derived fields
        'Vote_Share_Deviation': np.random.normal(0, 0.1, n_precincts),
    })
    
    # Add anomaly flags
    sample_data['Shpilkin_Anomaly_Flag'] = (sample_data['Shpilkin_Anomaly_Score'] > 2).astype(int)
    sample_data['ML_Anomaly_Flag'] = (sample_data['ML_Composite_Score'] > 0.7).astype(int)
    sample_data['Composite_Anomaly_Flag'] = (sample_data['Composite_Anomaly_Score'] > 0.6).astype(int)
    
    # Create visualizations
    visualizer = ComprehensiveVisualizer()
    plots = visualizer.create_analysis_dashboard(sample_data)
    
    print(f"Created visualization dashboard with {len(plots)} plots:")
    for plot_name in plots.keys():
        print(f"  - {plot_name}")


if __name__ == "__main__":
    main()
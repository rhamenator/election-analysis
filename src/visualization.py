"""Plotly and optional Folium visualizations for validated analysis results."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import load_config


class ShpilkinPlotter:
    """Compatibility name for turnout/share exploratory plots."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["visualization"]["colors"]
        supplied = (config or {}).get("colors", config or {})
        self.colors = {**defaults, **supplied}

    @staticmethod
    def _validate(frame: pd.DataFrame, candidate_column: str) -> str:
        if not candidate_column.startswith("Candidate_Share__") or candidate_column not in frame:
            raise ValueError("candidate_column must be a configured candidate share")
        if "Calculated_Turnout_Percent" in frame:
            return "Calculated_Turnout_Percent"
        if "Reported_Turnout_Percent" in frame:
            return "Reported_Turnout_Percent"
        raise ValueError("A mapped turnout field is required")

    def create_turnout_scatter(
        self,
        frame: pd.DataFrame,
        candidate_column: str,
        anomaly_column: str | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        turnout = self._validate(frame, candidate_column)
        work = frame.dropna(subset=[turnout, candidate_column]).copy()
        if work.empty:
            raise ValueError("No valid turnout/share observations to plot")
        if anomaly_column and anomaly_column in work:
            colors = np.where(
                work[anomaly_column].fillna(False).astype(bool),
                self.colors["flagged"],
                self.colors["normal"],
            )
        else:
            colors = np.repeat(self.colors["normal"], len(work))
        custom = work.reindex(columns=["Jurisdiction", "Precinct"]).fillna("").to_numpy()
        figure = go.Figure(
            go.Scatter(
                x=work[turnout],
                y=work[candidate_column],
                mode="markers",
                name="Precincts",
                marker={"color": colors, "size": 7, "opacity": 0.7},
                customdata=custom,
                hovertemplate=(
                    "Turnout: %{x:.2f}%<br>Share: %{y:.2%}<br>"
                    "Jurisdiction: %{customdata[0]}<br>Precinct: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        if "Turnout_Baseline_Expected_Share" in work:
            expected = work.sort_values(turnout)
            figure.add_trace(
                go.Scatter(
                    x=expected[turnout],
                    y=expected["Turnout_Baseline_Expected_Share"],
                    mode="lines",
                    name="Leave-one-out expected share",
                    line={"color": self.colors["expected"]},
                )
            )
        figure.update_layout(
            title="Exploratory turnout versus candidate share",
            xaxis_title="Turnout (%)",
            yaxis_title="Candidate share of valid contest votes",
            yaxis_tickformat=".0%",
        )
        if save_path:
            figure.write_html(save_path)
        return figure

    def create_residual_plot(
        self,
        frame: pd.DataFrame,
        candidate_column: str,
        save_path: str | None = None,
    ) -> go.Figure:
        turnout = self._validate(frame, candidate_column)
        required = [
            "Turnout_Share_Residual",
            "Turnout_Share_Prediction_Lower",
            "Turnout_Share_Prediction_Upper",
            "Turnout_Baseline_Expected_Share",
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Residual diagnostics are missing: {missing}")
        work = frame.dropna(subset=[turnout, *required]).sort_values(turnout)
        upper_residual = (
            work["Turnout_Share_Prediction_Upper"] - work["Turnout_Baseline_Expected_Share"]
        )
        lower_residual = (
            work["Turnout_Share_Prediction_Lower"] - work["Turnout_Baseline_Expected_Share"]
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=work[turnout],
                y=upper_residual,
                mode="lines",
                name="Prediction interval",
                line={"color": self.colors["expected"], "dash": "dot"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=work[turnout],
                y=lower_residual,
                mode="lines",
                name="Prediction interval",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(0,158,115,0.15)",
                line={"color": self.colors["expected"], "dash": "dot"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=work[turnout],
                y=work["Turnout_Share_Residual"],
                mode="markers",
                name="Leave-one-out residual",
                marker={"color": self.colors["normal"], "size": 7},
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="#333333")
        figure.update_layout(
            title="Turnout/share leave-one-out residuals",
            xaxis_title="Turnout (%)",
            yaxis_title="Observed minus expected share",
            yaxis_tickformat=".1%",
        )
        if save_path:
            figure.write_html(save_path)
        return figure

    def create_turnout_histogram(
        self,
        frame: pd.DataFrame,
        candidate_vote_columns: Iterable[str],
        *,
        bins: int = 25,
    ) -> go.Figure:
        """Aggregate candidate votes into turnout ranges, matching ETA's histogram semantics."""
        turnout = self._validate(
            frame,
            next(
                (
                    str(column)
                    for column in frame.columns
                    if str(column).startswith("Candidate_Share__")
                ),
                "",
            ),
        )
        columns = [column for column in candidate_vote_columns if column in frame]
        if not columns or bins < 2:
            raise ValueError("At least one candidate vote column and two bins are required")
        work = (
            frame[[turnout, *columns]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=[turnout])
        )
        if work.empty or work[turnout].nunique() < 2:
            raise ValueError("Turnout histogram requires varying valid turnout observations")
        turnout_values = np.asarray(work[turnout], dtype=float)
        edges = np.linspace(turnout_values.min(), turnout_values.max(), bins + 1)
        categories = pd.Series(
            pd.cut(
                pd.Series(turnout_values),
                [float(edge) for edge in edges],
                include_lowest=True,
                duplicates="drop",
            ),
            index=work.index,
        )
        centers = np.array([interval.mid for interval in categories.cat.categories])
        figure = go.Figure()
        for column in columns:
            totals = (
                work.groupby(categories, observed=False)[column]
                .sum()
                .reindex(categories.cat.categories)
            )
            figure.add_trace(go.Bar(x=centers, y=totals, name=column, opacity=0.72))
        figure.update_layout(
            title="Candidate votes by precinct turnout range",
            xaxis_title="Turnout range (%) — not time",
            yaxis_title="Candidate votes in precincts within range",
            barmode="overlay",
        )
        return figure

    def create_vote_share_by_count(
        self,
        frame: pd.DataFrame,
        candidate_share: str,
        candidate_votes: str,
        expected_column: str | None = None,
    ) -> go.Figure:
        required = [candidate_share, candidate_votes]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Vote-share-by-count fields are missing: {missing}")
        work = frame.dropna(subset=required).sort_values(candidate_votes)
        if work.empty:
            raise ValueError("No vote-count/share observations are available")
        figure = go.Figure(
            go.Scatter(
                x=work[candidate_votes],
                y=work[candidate_share],
                mode="markers",
                name="Precincts",
                marker={"color": self.colors["normal"], "opacity": 0.7},
            )
        )
        if expected_column and expected_column in work:
            figure.add_trace(
                go.Scatter(
                    x=work[candidate_votes],
                    y=work[expected_column],
                    mode="lines",
                    name="Descriptive linear trend",
                    line={"color": self.colors["expected"]},
                )
            )
        figure.update_layout(
            title="Candidate share by candidate vote count",
            xaxis_title="Candidate votes at precinct — not time",
            yaxis_title="Candidate share of valid contest votes",
            yaxis_tickformat=".0%",
        )
        return figure

    @staticmethod
    def create_down_ballot_difference(
        frame: pd.DataFrame, presidential_votes: str, difference_percent: str
    ) -> go.Figure:
        if presidential_votes not in frame or difference_percent not in frame:
            raise ValueError("Down-ballot plot fields are unavailable")
        work = frame.dropna(subset=[presidential_votes, difference_percent])
        if work.empty:
            raise ValueError("No valid down-ballot comparisons are available")
        figure = go.Figure(
            go.Scatter(
                x=work[presidential_votes],
                y=work[difference_percent],
                mode="markers",
                marker={"opacity": 0.7},
                name="Precincts",
            )
        )
        figure.add_hline(y=0, line_dash="dash")
        figure.update_layout(
            title="Presidential versus same-party down-ballot difference",
            xaxis_title="Presidential candidate votes",
            yaxis_title="Difference as % of presidential votes",
        )
        return figure


class GeospatialVisualizer:
    """Precinct marker maps; polygon choropleths are only created from polygons."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        defaults = load_config(None)["visualization"]
        config = config or {}
        self.map_config = {**defaults["map"], **config.get("map", {})}
        self.colors = {**defaults["colors"], **config.get("colors", {})}

    @staticmethod
    def _valid(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        required = ["Latitude", "Longitude", value_column]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Marker map fields are missing: {missing}")
        work = frame.dropna(subset=required)
        if work.empty:
            raise ValueError("No records have both valid coordinates and the selected value")
        return work

    def create_marker_map(self, frame: pd.DataFrame, value_column: str) -> go.Figure:
        work = self._valid(frame, value_column)
        figure = go.Figure(
            go.Scattermap(
                lat=work["Latitude"],
                lon=work["Longitude"],
                mode="markers",
                marker={
                    "size": self.map_config["marker_size"],
                    "color": work[value_column],
                    "colorscale": "Viridis",
                    "showscale": True,
                },
                text=work["Jurisdiction"].astype(str) + " — " + work["Precinct"].astype(str),
                hovertemplate="%{text}<br>Value: %{marker.color:.4f}<extra></extra>",
            )
        )
        figure.update_layout(
            title="Precinct marker map (not a choropleth)",
            map={
                "style": "open-street-map",
                "center": {"lat": work["Latitude"].mean(), "lon": work["Longitude"].mean()},
                "zoom": self.map_config["default_zoom"],
            },
            margin={"r": 0, "t": 45, "l": 0, "b": 0},
        )
        return figure

    def create_folium_map(
        self,
        frame: pd.DataFrame,
        anomaly_column: str,
        center_coords: tuple[float, float] | None = None,
    ) -> Any:
        try:
            import folium
        except ImportError as exc:
            raise ImportError("Install the dashboard dependency group for Folium maps") from exc
        work = self._valid(frame, anomaly_column)
        center = center_coords or (float(work["Latitude"].mean()), float(work["Longitude"].mean()))
        marker_map = folium.Map(
            location=center,
            zoom_start=self.map_config["default_zoom"],
            tiles=self.map_config["tile_layer"],
        )
        for row in work.itertuples(index=False):
            score = float(getattr(row, anomaly_column))
            folium.CircleMarker(
                location=[
                    float(cast(Any, row.Latitude)),
                    float(cast(Any, row.Longitude)),
                ],
                radius=self.map_config["marker_size"],
                color=self.colors["flagged"] if score >= 0.8 else self.colors["normal"],
                fill=True,
                tooltip=f"{row.Jurisdiction} — {row.Precinct}: {score:.3f}",
            ).add_to(marker_map)
        return marker_map

    def create_heatmap(
        self, frame: pd.DataFrame, value_column: str, save_path: str | None = None
    ) -> go.Figure:
        work = self._valid(frame, value_column)
        figure = go.Figure(
            go.Densitymap(
                lat=work["Latitude"],
                lon=work["Longitude"],
                z=work[value_column],
                radius=10,
                colorscale="Viridis",
            )
        )
        figure.update_layout(
            map={
                "style": "open-street-map",
                "center": {"lat": work["Latitude"].mean(), "lon": work["Longitude"].mean()},
                "zoom": self.map_config["default_zoom"],
            }
        )
        if save_path:
            figure.write_html(save_path)
        return figure

    def create_choropleth_by_county(self, *_: Any, **__: Any) -> go.Figure:
        raise ValueError(
            "A county choropleth requires county polygon geometry. Use create_marker_map for "
            "coordinate-only data."
        )


class StatisticalPlotter:
    """Generic distribution, correlation, and flag summaries."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def create_distribution_plots(
        self, frame: pd.DataFrame, columns: Iterable[str], save_path: str | None = None
    ) -> go.Figure:
        selected = [
            column
            for column in columns
            if column in frame and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not selected:
            raise ValueError("No numeric columns were supplied for distribution plots")
        width = min(3, len(selected))
        rows = (len(selected) + width - 1) // width
        figure = make_subplots(rows=rows, cols=width, subplot_titles=selected)
        for index, column in enumerate(selected):
            figure.add_trace(
                go.Histogram(x=frame[column].dropna(), name=column, showlegend=False),
                row=index // width + 1,
                col=index % width + 1,
            )
        figure.update_layout(title="Distribution diagnostics", height=300 * rows)
        if save_path:
            figure.write_html(save_path)
        return figure

    def create_correlation_heatmap(
        self, frame: pd.DataFrame, columns: Iterable[str], save_path: str | None = None
    ) -> go.Figure:
        selected = [column for column in columns if column in frame]
        if len(selected) < 2:
            raise ValueError("At least two result columns are required for correlation")
        correlation = frame[selected].corr(numeric_only=True)
        figure = go.Figure(
            go.Heatmap(z=correlation.to_numpy(), x=correlation.columns, y=correlation.index, zmid=0)
        )
        if save_path:
            figure.write_html(save_path)
        return figure

    def create_anomaly_summary_plot(
        self, frame: pd.DataFrame, columns: Iterable[str], save_path: str | None = None
    ) -> go.Figure:
        selected = [column for column in columns if column in frame]
        if not selected:
            raise ValueError("No flag columns were supplied")
        counts = [int(frame[column].fillna(False).astype(bool).sum()) for column in selected]
        figure = go.Figure(go.Bar(x=selected, y=counts))
        figure.update_layout(title="Flag counts by independent method", yaxis_title="Records")
        if save_path:
            figure.write_html(save_path)
        return figure


class ComprehensiveVisualizer:
    """Build only plots supported by columns actually produced in a run."""

    def __init__(self, config_path: str | None = "config.yaml") -> None:
        self.config = load_config(config_path)["visualization"]
        self.shpilkin = ShpilkinPlotter(self.config)
        self.geospatial = GeospatialVisualizer(self.config)
        self.statistical = StatisticalPlotter(self.config)

    def create_analysis_dashboard(
        self,
        frame: pd.DataFrame,
        output_dir: str | Path = "plots",
        candidate_column: str | None = None,
    ) -> dict[str, go.Figure]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        candidate = candidate_column or next(
            (
                str(column)
                for column in frame.columns
                if str(column).startswith("Candidate_Share__")
            ),
            None,
        )
        plots: dict[str, go.Figure] = {}
        if candidate:
            plots["turnout_share"] = self.shpilkin.create_turnout_scatter(
                frame,
                candidate,
                "Turnout_Share_Flag" if "Turnout_Share_Flag" in frame else None,
                str(output / "turnout_share.html"),
            )
            if "Turnout_Share_Residual" in frame:
                plots["turnout_share_residuals"] = self.shpilkin.create_residual_plot(
                    frame, candidate, str(output / "turnout_share_residuals.html")
                )
        numeric = [str(column) for column in frame.select_dtypes(include=np.number).columns[:6]]
        if numeric:
            plots["distributions"] = self.statistical.create_distribution_plots(
                frame, numeric, str(output / "distributions.html")
            )
        map_value = next(
            (column for column in ("IF_Anomaly_Score", candidate) if column and column in frame),
            None,
        )
        if map_value and {"Latitude", "Longitude"}.issubset(frame):
            plots["precinct_marker_map"] = self.geospatial.create_marker_map(frame, map_value)
            plots["precinct_marker_map"].write_html(output / "precinct_marker_map.html")
        return plots


def main() -> None:
    from .data_ingestion import ElectionDataIngester
    from .sample_data import generalized_sample_data
    from .statistical_models import StatisticalAnomalyDetector

    ingestion = ElectionDataIngester().process(
        generalized_sample_data().to_csv(index=False).encode()
    )
    run = StatisticalAnomalyDetector().run(
        ingestion.data,
        ingestion.schema.candidates[0].share_column,
        candidate_vote_columns=ingestion.schema.candidate_columns,
        methods=["turnout_share"],
    )
    plots = ComprehensiveVisualizer().create_analysis_dashboard(
        run.data, candidate_column=ingestion.schema.candidates[0].share_column
    )
    print(f"Created {len(plots)} tested plot objects")


if __name__ == "__main__":
    main()

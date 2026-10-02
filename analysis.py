import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Common chart styling
CHART_TEMPLATE = "plotly_dark"

COLORS = {
    "TV": "#818CF8",
    "Radio": "#22D3EE",
    "Newspaper": "#F472B6",
    "Sales": "#34D399"
}


def get_summary(df):
    """Return basic dataset statistics."""

    return {
        "records": len(df),
        "average_sales": df["Sales"].mean(),
        "highest_sales": df["Sales"].max(),
        "total_ad_spend": (
            df["TV"] + df["Radio"] + df["Newspaper"]
        ).sum()
    }


def advertising_distribution(df):
    """Show total advertising expenditure by channel."""

    channels = ["TV", "Radio", "Newspaper"]

    totals = df[channels].sum().reset_index()
    totals.columns = ["Channel", "Expenditure"]

    fig = px.bar(
        totals,
        x="Channel",
        y="Expenditure",
        color="Channel",
        color_discrete_map=COLORS,
        title="Advertising Expenditure by Channel"
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        showlegend=False,
        title_x=0.03,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig


def sales_relationship(df, channel="TV"):
    """Visualize advertising expenditure versus sales."""

    if channel not in ["TV", "Radio", "Newspaper"]:
        raise ValueError("Invalid advertising channel")

    fig = px.scatter(
        df,
        x=channel,
        y="Sales",
        color_discrete_sequence=[COLORS[channel]],
        trendline="ols",
        title=f"{channel} Advertising vs Sales",
        opacity=0.75
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        title_x=0.03,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig


def correlation_heatmap(df):
    """Display correlations between numeric variables."""

    columns = ["TV", "Radio", "Newspaper", "Sales"]
    correlation = df[columns].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
        aspect="auto"
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        title_x=0.03,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig


def actual_vs_predicted(results):
    """Compare actual sales with model predictions."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["Actual"],
            y=results["Predicted"],
            mode="markers",
            name="Predictions",
            marker=dict(
                color="#818CF8",
                size=9,
                opacity=0.8
            )
        )
    )

    lower = min(
        results["Actual"].min(),
        results["Predicted"].min()
    )

    upper = max(
        results["Actual"].max(),
        results["Predicted"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color="#34D399",
                dash="dash",
                width=2
            )
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Sales",
        template=CHART_TEMPLATE,
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
        title_x=0.03,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig


def model_comparison(cv_scores):
    """Compare models using cross-validation R²."""

    comparison = pd.DataFrame({
        "Model": list(cv_scores.keys()),
        "R² Score": list(cv_scores.values())
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="R² Score",
        color="Model",
        color_discrete_sequence=[
            "#818CF8",
            "#22D3EE"
        ],
        title="Model Comparison (Cross-Validation)"
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        showlegend=False,
        title_x=0.03,
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig
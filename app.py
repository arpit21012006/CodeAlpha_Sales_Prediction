from pathlib import Path
import html

import pandas as pd
import streamlit as st

from preprocessing import get_dataset
from model import train_models, predict_sales
from analysis import (
    get_summary,
    advertising_distribution,
    sales_relationship,
    correlation_heatmap,
    actual_vs_predicted,
    model_comparison
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SalesIQ | Sales Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSS_FILE = BASE_DIR / "style.css"

if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# --------------------------------------------------
# Load data and train model
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset():
    return get_dataset()


@st.cache_resource(show_spinner=False)
def load_model():
    return train_models()


try:
    with st.spinner("Preparing sales analytics..."):
        df = load_dataset()
        trained = load_model()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


model = trained["model"]
model_name = trained["model_name"]
metrics = trained["metrics"]
results = trained["results"]
cv_scores = trained["cv_scores"]

summary = get_summary(df)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📈 SalesIQ")
    st.caption("SALES INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### Project Overview")

    st.write("**Project:** Sales Prediction")
    st.write("**Domain:** Data Science")
    st.write("**Algorithm:**", model_name)
    st.write("**Dataset:** Advertising")

    st.divider()

    st.markdown("### Model Performance")

    st.metric(
        "Test R² Score",
        f"{metrics['R2']:.3f}"
    )

    st.metric(
        "Mean Absolute Error",
        f"{metrics['MAE']:.2f}"
    )

    st.divider()

    st.info(
        "Sales predictions are estimates based on "
        "historical advertising data."
    )

    st.caption("Built with Python & Streamlit")


# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-eyebrow">
            MACHINE LEARNING • ANALYTICS
        </div>
        <div class="hero-title">
            Sales Intelligence Dashboard
        </div>
        <div class="hero-description">
            Explore advertising performance, understand
            sales patterns, and estimate sales using
            machine learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Overview metrics
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{summary['records']:,}"
    )

with col2:
    st.metric(
        "Average Sales",
        f"{summary['average_sales']:.2f}"
    )

with col3:
    st.metric(
        "Highest Sales",
        f"{summary['highest_sales']:.2f}"
    )

with col4:
    st.metric(
        "Model R²",
        f"{metrics['R2']:.3f}"
    )


st.write("")


# --------------------------------------------------
# Dashboard navigation
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Sales Prediction",
        "📊 Data Analytics",
        "🤖 Model Performance",
        "📁 Dataset Explorer"
    ]
)


# ==================================================
# TAB 1 — SALES PREDICTION
# ==================================================

with tab1:

    st.markdown("## Predict Future Sales")

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Enter advertising budgets to estimate sales '
        'using the trained regression model.'
        '</p>',
        unsafe_allow_html=True
    )

    st.info(
        "Use the same advertising budget units as "
        "the original dataset. Predicted sales are "
        "expressed in the dataset's sales units."
    )

    with st.form("sales_prediction_form"):

        left, middle, right = st.columns(3)

        with left:
            tv_budget = st.number_input(
                "📺 TV Advertising Budget",
                min_value=0.0,
                max_value=500.0,
                value=150.0,
                step=5.0,
                help="Advertising expenditure on TV."
            )

        with middle:
            radio_budget = st.number_input(
                "📻 Radio Advertising Budget",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                help="Advertising expenditure on radio."
            )

        with right:
            newspaper_budget = st.number_input(
                "📰 Newspaper Advertising Budget",
                min_value=0.0,
                max_value=150.0,
                value=30.0,
                step=1.0,
                help="Advertising expenditure on newspapers."
            )

        submitted = st.form_submit_button(
            "✨ Predict Sales",
            use_container_width=True
        )

    if submitted:

        prediction = predict_sales(
            model,
            tv_budget,
            radio_budget,
            newspaper_budget
        )

        total_budget = (
            tv_budget
            + radio_budget
            + newspaper_budget
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Estimated Sales
                </div>
                <div class="prediction-value">
                    {prediction:,.2f}
                </div>
                <div class="prediction-note">
                    Prediction generated using
                    {html.escape(model_name)}.
                    Actual sales may vary.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Total Advertising Budget",
                f"{total_budget:,.2f}"
            )

        with result_col2:
            st.metric(
                "Predicted Sales",
                f"{prediction:,.2f}"
            )

        st.caption(
            "This is a statistical estimate, not "
            "a guaranteed business outcome. "
            "Predictions outside the training data "
            "range may be less reliable."
        )

    st.divider()

    st.markdown("### How prediction works")

    st.markdown(
        """
        The model learns relationships between
        advertising expenditure and sales from
        historical records.

        Enter the advertising budgets for **TV**,
        **Radio**, and **Newspaper**, then select
        **Predict Sales** to generate an estimate.
        """
    )


# ==================================================
# TAB 2 — DATA ANALYTICS
# ==================================================

with tab2:

    st.markdown("## Advertising Analytics")

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Explore spending patterns and their '
        'relationship with sales.'
        '</p>',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        advertising_distribution(df),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Advertising Channel Analysis")

    selected_channel = st.selectbox(
        "Select Advertising Channel",
        ["TV", "Radio", "Newspaper"]
    )

    st.plotly_chart(
        sales_relationship(df, selected_channel),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Feature Correlations")

    st.plotly_chart(
        correlation_heatmap(df),
        use_container_width=True
    )

    st.caption(
        "Correlation describes statistical association; "
        "it does not establish causation."
    )


# ==================================================
# TAB 3 — MODEL PERFORMANCE
# ==================================================

with tab3:

    st.markdown("## Machine Learning Performance")

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Evaluate prediction quality and compare '
        'regression algorithms.'
        '</p>',
        unsafe_allow_html=True
    )

    st.success(
        f"Selected Model: {model_name}"
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "R² Score",
            f"{metrics['R2']:.3f}"
        )

    with metric2:
        st.metric(
            "MAE",
            f"{metrics['MAE']:.3f}"
        )

    with metric3:
        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.3f}"
        )

    st.caption(
        "Metrics above are calculated on the held-out "
        "test dataset. Lower MAE and RMSE indicate "
        "smaller prediction errors."
    )

    st.divider()

    st.plotly_chart(
        actual_vs_predicted(results),
        use_container_width=True
    )

    st.divider()

    st.plotly_chart(
        model_comparison(cv_scores),
        use_container_width=True
    )

    st.caption(
        "Models were compared using five-fold "
        "cross-validation on the training data. "
        "The model with the higher mean validation "
        "R² was selected before test evaluation."
    )

    st.markdown("### Prediction Comparison")

    comparison_table = results.copy()

    comparison_table["Absolute Error"] = (
        comparison_table["Actual"]
        - comparison_table["Predicted"]
    ).abs()

    st.dataframe(
        comparison_table.round(3),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 4 — DATASET EXPLORER
# ==================================================

with tab4:

    st.markdown("## Explore the Dataset")

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Inspect the cleaned advertising dataset '
        'used for machine learning.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Descriptive Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Dataset Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Rows", len(df))

    with info2:
        st.metric("Columns", len(df.columns))

    with info3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_advertising.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="dashboard-footer">
        SalesIQ • Sales Prediction Dashboard
        <br>
        Built using Python, Scikit-learn,
        Plotly and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
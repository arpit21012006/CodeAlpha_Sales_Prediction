# SalesIQ - Sales Prediction Using Python

A machine learning project that predicts sales based on advertising expenditure across TV, Radio, and Newspaper channels.

Developed as part of the CodeAlpha Data Science Internship.

## Project Overview

SalesIQ is an interactive sales prediction dashboard built using Python and Streamlit.

The project uses historical advertising data to analyze spending patterns, train regression models, and estimate sales from advertising budgets.

## Features

- Sales prediction using machine learning
- Interactive Streamlit dashboard
- Advertising expenditure analysis
- TV, Radio, and Newspaper sales relationships
- Correlation heatmap
- Linear Regression and Random Forest comparison
- Model evaluation using R², MAE, and RMSE
- Actual vs Predicted visualization
- Dataset exploration and CSV download
- Custom CSS styling

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Statsmodels
- HTML and CSS

## Dataset

The project uses the Advertising dataset containing 200 records.

Input features:
- TV advertising expenditure
- Radio advertising expenditure
- Newspaper advertising expenditure

Target variable:
- Sales

The unnecessary index column is removed during preprocessing.

## Project Structure

```text
CodeAlpha_Sales_Prediction/
│
├── app.py
├── preprocessing.py
├── model.py
├── analysis.py
├── style.css
├── Advertising.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Approach

1. Load and clean the dataset.
2. Separate input features and target variable.
3. Split data into training and testing sets.
4. Compare Linear Regression and Random Forest using five-fold cross-validation on training data.
5. Select the model with the higher average validation R².
6. Evaluate the selected model on the held-out test set.
7. Generate sales predictions using user-provided advertising budgets.

## Model Evaluation

The application displays:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Actual vs Predicted Sales
- Cross-validation model comparison

Evaluation results are generated when the application runs.

## Installation

Navigate to the project directory:

```bash
cd CodeAlpha_Sales_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## How to Use

1. Open the Sales Prediction tab.
2. Enter TV, Radio, and Newspaper advertising budgets.
3. Click Predict Sales.
4. View the estimated sales.
5. Explore analytics, model performance, and dataset information using the other tabs.

## Limitations

- Predictions are based on a relatively small historical dataset.
- Advertising expenditure is not the only factor influencing sales.
- Predictions outside the training data range may be less reliable.
- Model predictions are estimates rather than guaranteed outcomes.

## Internship
ARPIT SHARMA 

CodeAlpha Data Science Internship

Project: Sales Prediction Using Python
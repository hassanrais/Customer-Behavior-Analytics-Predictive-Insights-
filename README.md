# Online Shoppers Intention Analysis Dashboard

This is an interactive dashboard for analyzing online shoppers' behavior and purchase intentions.

## Features

- Data Overview: Basic statistics and visualizations
- Classification Analysis: Multiple ML models for purchase prediction
- Clustering Analysis: Customer segmentation using K-Means and Hierarchical Clustering
- Association Rules: Pattern discovery using Apriori and FP-Growth algorithms

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have the dataset file `online_shoppers_intention.csv` in the same directory as `app.py`

## Running the Application

Run the following command in your terminal:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Use the sidebar to navigate between different analysis sections
2. In each section, you can:
   - Adjust parameters using sliders and dropdowns
   - View visualizations and results
   - Download generated reports and plots

## Data Requirements

The application expects a CSV file named `online_shoppers_intention.csv` with the following columns:
- Administrative_Duration
- Informational_Duration
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- Month
- VisitorType
- Weekend
- Revenue 
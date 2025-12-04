import os
import click
import pickle
import numpy as np
import pandas as pd
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_OUTPUT_PREFIX = '../results/model_results'
DEFAULT_SEED = 522

NUMERIC_FEATURES = [
    'Length', 'Diameter', 'Height', 
    'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'
]
CATEGORICAL_FEATURES = ['Sex']
TARGET_COLUMN = 'Rings'


def create_preprocessor():
    """Create preprocessing pipeline for numeric and categorical features."""
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(drop='if_binary', sparse_output=False), CATEGORICAL_FEATURES)
    )
    return preprocessor


def train_models(X, y, seed):
    """Train Ridge Regression and Random Forest models."""
    models = {}
    
    # Ridge Regression
    ridge_pipeline = make_pipeline(
        create_preprocessor(),
        Ridge(alpha=1.0, random_state=seed)
    )
    ridge_pipeline.fit(X, y)
    models['Ridge Regression'] = ridge_pipeline
    
    # Random Forest
    rf_pipeline = make_pipeline(
        create_preprocessor(),
        RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1
        )
    )
    rf_pipeline.fit(X, y)
    models['Random Forest'] = rf_pipeline
    
    return models


def evaluate_models(models, X, y):
    """Evaluate models and return metrics DataFrame."""
    results = []
    for name, model in models.items():
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        results.append({
            'Model': name,
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4)
        })
    return pd.DataFrame(results)


def create_predictions_df(models, X, y):
    """Create DataFrame with actual vs predicted values."""
    predictions = []
    for name, model in models.items():
        y_pred = model.predict(X)
        for actual, predicted in zip(y, y_pred):
            predictions.append({
                'Model': name,
                'Actual': actual,
                'Predicted': predicted
            })
    return pd.DataFrame(predictions)


def create_actual_vs_predicted_chart(predictions_df):
    """Create scatter plot of actual vs predicted values."""
    chart = alt.Chart(predictions_df).mark_circle(opacity=0.5).encode(
        x=alt.X('Actual:Q', title='Actual Rings'),
        y=alt.Y('Predicted:Q', title='Predicted Rings'),
        color=alt.Color('Model:N', title='Model'),
        tooltip=['Model', 'Actual', 'Predicted']
    ).properties(
        width=300,
        height=300
    ).facet(
        column=alt.Column('Model:N', title=None)
    ).properties(
        title='Actual vs Predicted Rings by Model'
    )
    return chart


def create_model_comparison_chart(metrics_df):
    """Create bar chart comparing model performance."""
    melted = metrics_df.melt(
        id_vars=['Model'], 
        value_vars=['RMSE', 'R2_Score'],
        var_name='Metric',
        value_name='Value'
    )
    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X('Model:N', title='Model'),
        y=alt.Y('Value:Q', title='Value'),
        color=alt.Color('Model:N', legend=None),
        column=alt.Column('Metric:N', title=None)
    ).properties(
        width=200,
        height=300,
        title='Model Performance Comparison'
    )
    return chart


def create_residuals_chart(predictions_df):
    """Create residuals plot."""
    residuals_df = predictions_df.copy()
    residuals_df['Residual'] = residuals_df['Predicted'] - residuals_df['Actual']
    
    chart = alt.Chart(residuals_df).mark_circle(opacity=0.5).encode(
        x=alt.X('Predicted:Q', title='Predicted Rings'),
        y=alt.Y('Residual:Q', title='Residual (Predicted - Actual)'),
        color=alt.Color('Model:N', title='Model'),
        tooltip=['Model', 'Actual', 'Predicted', 'Residual']
    ).properties(
        width=300,
        height=300
    ).facet(
        column=alt.Column('Model:N', title=None)
    ).properties(
        title='Residuals Plot by Model'
    )
    return chart


def save_chart(chart, path):
    """Save chart to file, falling back to HTML if PNG fails."""
    try:
        chart.save(path, scale_factor=2)
    except ValueError:
        html_path = path.replace('.png', '.html')
        chart.save(html_path)


def train_and_save_results(train_path, output_prefix, seed):
    """
    Train models on the data and save results as figures and tables.
    """
    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Load data
    df = pd.read_csv(train_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    # Train models
    models = train_models(X, y, seed)
    
    # Evaluate models
    metrics_df = evaluate_models(models, X, y)
    predictions_df = create_predictions_df(models, X, y)
    
    # Save models
    model_path = f"{output_prefix}_models.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    # Save metrics table
    metrics_path = f"{output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save predictions table
    predictions_path = f"{output_prefix}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save figures
    save_chart(
        create_actual_vs_predicted_chart(predictions_df),
        f"{output_prefix}_actual_vs_predicted.png"
    )
    save_chart(
        create_model_comparison_chart(metrics_df),
        f"{output_prefix}_model_comparison.png"
    )
    save_chart(
        create_residuals_chart(predictions_df),
        f"{output_prefix}_residuals.png"
    )


@click.command()
@click.option(
    '--train-path',
    type=str,
    default=DEFAULT_TRAIN_PATH,
    help='Path to training data CSV file'
)
@click.option(
    '--output-prefix',
    type=str,
    default=DEFAULT_OUTPUT_PREFIX,
    help='Path/filename prefix for output figures and tables'
)
@click.option(
    '--seed',
    type=int,
    default=DEFAULT_SEED,
    help='Random seed for reproducibility'
)
def main(train_path, output_prefix, seed):
    """Train models on processed data and save results as figures and tables."""
    train_and_save_results(train_path, output_prefix, seed)


if __name__ == "__main__":
    main()

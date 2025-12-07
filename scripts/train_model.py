import os
import click
import pickle
import numpy as np
import pandas as pd
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_TEST_PATH = '../data/processed/abalone_test.csv'
DEFAULT_OUTPUT_PREFIX = '../results/model/model_results'
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
    """Train Linear Regression, Random Forest, and SVR models."""
    models = {}
    
    # Linear Regression (Baseline)
    lr_pipeline = make_pipeline(
        create_preprocessor(),
        LinearRegression()
    )
    lr_pipeline.fit(X, y)
    models['Linear Regression'] = lr_pipeline
    
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
    
    # SVR (RBF Kernel)
    svr_pipeline = make_pipeline(
        create_preprocessor(),
        SVR(kernel='rbf', C=1.0, epsilon=0.1)
    )
    svr_pipeline.fit(X, y)
    models['SVR (RBF Kernel)'] = svr_pipeline
    
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


def create_lr_scatter_chart(predictions_df, metrics_df):
    """Create scatter plot for Linear Regression only with perfect prediction line."""
    # Filter for Linear Regression only
    lr_predictions = predictions_df[predictions_df['Model'] == 'Linear Regression'].copy()
    
    # Get metrics for subtitle
    lr_metrics = metrics_df[metrics_df['Model'] == 'Linear Regression'].iloc[0]
    lr_r2 = lr_metrics['R2_Score']
    lr_rmse = lr_metrics['RMSE']
    
    # Create scatter plot
    scatter = alt.Chart(lr_predictions).mark_circle(
        opacity=0.5,
        color='steelblue',
        size=60
    ).encode(
        x=alt.X('Actual:Q', 
                title='Actual Number of Rings',
                scale=alt.Scale(domain=[0, 30])),
        y=alt.Y('Predicted:Q', 
                title='Predicted Number of Rings',
                scale=alt.Scale(domain=[0, 30])),
        tooltip=['Actual', 'Predicted']
    )
    
    # Create perfect prediction line (y = x)
    line_data = pd.DataFrame({'x': [0, 30], 'y': [0, 30]})
    line = alt.Chart(line_data).mark_line(
        color='red',
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(
        x='x:Q',
        y='y:Q'
    )
    
    # Combine scatter and line
    chart = (scatter + line).properties(
        width=500,
        height=500,
        title=alt.TitleParams(
            text='Linear Regression: Actual vs Predicted Ring Count',
            fontSize=14,
            fontWeight='bold',
            subtitle=f'R² = {lr_r2:.3f}, RMSE = {lr_rmse:.2f} rings',
            subtitleFontSize=12
        )
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=13
    )
    
    return chart


def extract_lr_coefficients(lr_pipeline, numeric_features, categorical_features):
    """Extract coefficients from Linear Regression pipeline."""
    # Get the linear regression model from the pipeline
    lr_model = lr_pipeline.named_steps['linearregression']
    
    # Get the preprocessor to understand feature names after transformation
    preprocessor = lr_pipeline.named_steps['columntransformer']
    
    # Get one-hot encoder categories
    ohe = preprocessor.named_transformers_['onehotencoder']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    
    # Combine all feature names (numeric first, then categorical)
    all_feature_names = numeric_features + cat_feature_names
    
    # Get coefficients
    coefficients = lr_model.coef_
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute value of coefficient (descending)
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_df = coef_df.drop('Abs_Coefficient', axis=1)
    
    # Round coefficients
    coef_df['Coefficient'] = coef_df['Coefficient'].round(4)
    
    return coef_df


def save_chart(chart, path):
    """Save chart to PNG file."""
    chart.save(path)


def train_and_save_results(train_path, test_path, output_prefix, seed):
    """
    Train models on training data and evaluate on test data.
    Save results as figures and tables.
    """
    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Load training data
    train_df = pd.read_csv(train_path)
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    
    # Load test data
    test_df = pd.read_csv(test_path)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COLUMN]
    
    # Train models on training data
    models = train_models(X_train, y_train, seed)
    
    # Evaluate models on test data
    metrics_df = evaluate_models(models, X_test, y_test)
    predictions_df = create_predictions_df(models, X_test, y_test)
    
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
    
    # Save Linear Regression specific outputs
    save_chart(
        create_lr_scatter_chart(predictions_df, metrics_df),
        f"{output_prefix}_lr_scatter.png"
    )
    
    # Save Linear Regression coefficients table
    lr_coef_df = extract_lr_coefficients(
        models['Linear Regression'], 
        NUMERIC_FEATURES, 
        CATEGORICAL_FEATURES
    )
    lr_coef_df.to_csv(f"{output_prefix}_lr_coefficients.csv", index=False)


@click.command()
@click.option(
    '--train-path',
    type=str,
    default=DEFAULT_TRAIN_PATH,
    help='Path to training data CSV file'
)
@click.option(
    '--test-path',
    type=str,
    default=DEFAULT_TEST_PATH,
    help='Path to test data CSV file'
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
def main(train_path, test_path, output_prefix, seed):
    """Train models on training data and evaluate on test data."""
    train_and_save_results(train_path, test_path, output_prefix, seed)


if __name__ == "__main__":
    main()

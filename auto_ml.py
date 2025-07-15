#!/usr/bin/env python3
"""
Coal Production Prediction ML Pipeline using Scikit-learn AutoML
================================================================

This script implements an automated machine learning pipeline for predicting
coal mine production using scikit-learn. It loads data, performs feature engineering,
trains multiple models, selects the best one, and saves it for deployment.

Author: ML Expert
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def print_step_time(step_name, start_time):
    """Print the time taken for a step."""
    end_time = time.time()
    duration = end_time - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {step_name} completed in {duration:.2f} seconds")
    return end_time

def format_duration(duration_seconds):
    """Format duration in a human-readable format."""
    if duration_seconds < 60:
        return f"{duration_seconds:.2f} seconds"
    elif duration_seconds < 3600:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

def load_and_prepare_data(data_path='cleaned_coalpublic2015.csv'):
    """
    Load and prepare the coal production dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the cleaned coal production CSV file
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for ML pipeline
    """
    print("Loading coal production data...")
    
    # Try different possible paths
    possible_paths = [
        data_path,
        f'../data/{data_path}',
        f'data/{data_path}',
        f'./{data_path}'
    ]
    
    df = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, index_col='MSHA ID')
                print(f"Data loaded from: {path}")
                break
        except Exception as e:
            continue
    
    if df is None:
        raise FileNotFoundError(f"Could not find data file. Tried paths: {possible_paths}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def preprocess_data(df, target_col='log_production'):
    """
    Preprocess the data for machine learning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Target column name
        
    Returns:
    --------
    tuple
        Preprocessed features and target
    """
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define categorical and numerical columns
    categorical_cols = [
        'Mine_State', 'Mine_County', 'Mine_Status', 'Mine_Type',
        'Company_Type', 'Operation_Type', 'Union_Code', 'Coal_Supply_Region'
    ]
    
    numerical_cols = ['Average_Employees', 'Labor_Hours']
    
    # Drop non-feature columns
    columns_to_drop = ['Year', 'Mine_Name', 'Operating_Company', 'Operating_Company_Address', 'Production_(short_tons)']
    existing_cols_to_drop = [col for col in columns_to_drop if col in X.columns]
    if existing_cols_to_drop:
        X = X.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")
    
    # Handle categorical variables with one-hot encoding
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"Features after preprocessing: {X_processed.shape[1]}")
    print(f"Sample features: {list(X_processed.columns[:10])}")
    
    return X_processed, y

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Test set size ratio
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Training and testing sets
    """
    print(f"Splitting data into {100*(1-test_size):.0f}% train and {100*test_size:.0f}% test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def get_model_definitions():
    """
    Define all models to be tested in the AutoML pipeline.
    
    Returns:
    --------
    dict
        Dictionary of model names and their instances
    """
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Regression': SVR()
    }
    return models

def compare_models(X_train, y_train, cv_folds=5):
    """
    Compare multiple regression models using cross-validation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    pd.DataFrame
        Model comparison results sorted by R2 score
    """
    print(f"Comparing models with {cv_folds}-fold cross-validation...")
    
    models = get_model_definitions()
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            # Calculate metrics
            mean_r2 = cv_scores.mean()
            std_r2 = cv_scores.std()
            
            results.append({
                'Model': name,
                'Mean_R2': mean_r2,
                'Std_R2': std_r2,
                'Min_R2': cv_scores.min(),
                'Max_R2': cv_scores.max()
            })
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    # Create results dataframe and sort by R2
    results_df = pd.DataFrame(results).sort_values('Mean_R2', ascending=False)
    
    print("\nModel Comparison Results:")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    return results_df

def tune_best_model(X_train, y_train, best_model_name):
    """
    Tune hyperparameters for the best performing model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    best_model_name : str
        Name of the best performing model
        
    Returns:
    --------
    sklearn model
        Tuned model object
    """
    print(f"Tuning hyperparameters for {best_model_name}...")
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Extra Trees': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'Lasso Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'Elastic Net': {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        },
        'Support Vector Regression': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # Get the base model
    models = get_model_definitions()
    base_model = models[best_model_name]
    
    # Get parameter grid
    param_grid = param_grids.get(best_model_name, {})
    
    if not param_grid:
        print(f"No hyperparameter tuning defined for {best_model_name}, using default parameters.")
        return base_model
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R2 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model_performance(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate the model performance with various metrics.
    
    Parameters:
    -----------
    model : trained model object
        The model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    X_train : pd.DataFrame, optional
        Training features for training metrics
    y_train : pd.Series, optional
        Training target for training metrics
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    
    # Calculate test metrics
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    
    print("\nTest Set Performance:")
    print("=" * 30)
    print(f"R² Score: {test_r2:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Training metrics if provided
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        
        print("\nTraining Set Performance:")
        print("=" * 30)
        print(f"R² Score: {train_r2:.4f}")
        print(f"MAE: {train_mae:.4f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print(f"\nWarning: Possible overfitting detected!")
            print(f"Training R² - Test R² = {train_r2 - test_r2:.4f}")
    
    # Create prediction vs actual plot
    try:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Prediction vs Actual (R² = {test_r2:.3f})')
        
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"\nEvaluation plots saved as 'model_evaluation.png'")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    return {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse
    }

def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract and display feature importance from the model with simple plotting.
    
    Parameters:
    -----------
    model : trained model object
        The model to extract feature importance from
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    print(f"\nFeature Importance Analysis (Top {top_n}):")
    print("=" * 50)
    
    try:
        feature_importance = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting, etc.)
            importance_scores = model.feature_importances_
            importance_type = "Tree-based Feature Importance"
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores,
                'importance_percent': (importance_scores / importance_scores.sum()) * 100
            }).sort_values('importance', ascending=False)
            
            print(f"\n{importance_type}:")
            print(feature_importance.head(top_n).to_string(index=False, float_format='%.4f'))
            
            # Create simple feature importance plot
            create_simple_feature_importance_plot(feature_importance, top_n)
            
        elif hasattr(model, 'coef_'):
            # Linear models (Linear Regression, Ridge, Lasso, etc.)
            if len(model.coef_.shape) == 1:
                importance_scores = np.abs(model.coef_)
            else:
                importance_scores = np.abs(model.coef_[0])
            
            importance_type = "Linear Model Coefficients (Absolute Values)"
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores,
                'importance_percent': (importance_scores / importance_scores.sum()) * 100
            }).sort_values('importance', ascending=False)
            
            print(f"\n{importance_type}:")
            print(feature_importance.head(top_n).to_string(index=False, float_format='%.4f'))
            
            # Create simple feature importance plot
            create_simple_feature_importance_plot(feature_importance, top_n)
            
        else:
            print("Feature importance not available for this model type.")
            return None
        
        # Print summary statistics
        print(f"\nFeature Importance Summary:")
        print(f"Total features: {len(feature_importance)}")
        print(f"Top 5 features account for {feature_importance.head(5)['importance_percent'].sum():.1f}% of importance")
        print(f"Top 10 features account for {feature_importance.head(10)['importance_percent'].sum():.1f}% of importance")
        
        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"\nFeature importance saved as 'feature_importance.csv'")
        
        return feature_importance
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_feature_importance_plot(feature_importance, top_n=15):
    """
    Create a simple feature importance plot.
    
    Parameters:
    -----------
    feature_importance : pd.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to plot
    """
    try:
        # Create a simple feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), [f"{feat[:40]}..." if len(feat) > 40 else feat 
                                             for feat in top_features['feature']])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + row['importance'] * 0.01, i, 
                    f'{row["importance"]:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Add a subtle grid
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Improve layout
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved as 'feature_importance.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        import traceback
        traceback.print_exc()

def save_model(model, model_path='saved_models/best_model.sav', include_metadata=True):
    """
    Save the trained model in .sav format.
    
    Parameters:
    -----------
    model : trained model object
        The model to save
    model_path : str
        Path for the saved model file (including .sav extension)
    include_metadata : bool
        Whether to save additional metadata
        
    Returns:
    --------
    str
        Path to saved model file
    """
    print(f"Saving model as '{model_path}'...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model using joblib (recommended for scikit-learn models)
        joblib.dump(model, model_path)
        
        print(f"✓ Model saved successfully as '{model_path}'!")
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'model_type': type(model).__name__,
                'model_params': model.get_params() if hasattr(model, 'get_params') else None,
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_path': model_path
            }
            
            metadata_path = model_path.replace('.sav', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✓ Model metadata saved as '{metadata_path}'")
        
        return model_path
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def load_saved_model(model_path='saved_models/best_model.sav'):
    """
    Load a previously saved model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file (with .sav extension)
        
    Returns:
    --------
    Loaded model object
    """
    print(f"Loading saved model: {model_path}")
    
    try:
        loaded_model = joblib.load(model_path)
        
        print("Model loaded successfully!")
        
        # Try to load metadata
        try:
            metadata_path = model_path.replace('.sav', '_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"Model type: {metadata.get('model_type', 'Unknown')}")
            print(f"Saved on: {metadata.get('timestamp', 'Unknown')}")
        except:
            pass
        
        return loaded_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(model, new_data, feature_columns=None):
    """
    Make predictions on new data using the trained model.
    
    Parameters:
    -----------
    model : trained model object
        The model to use for predictions
    new_data : pd.DataFrame
        New data to make predictions on
    feature_columns : list, optional
        List of feature columns to use (in case of column mismatch)
        
    Returns:
    --------
    np.array
        Predictions array
    """
    print("Making predictions on new data...")
    
    try:
        # Ensure we have the right columns if specified
        if feature_columns is not None:
            missing_cols = set(feature_columns) - set(new_data.columns)
            if missing_cols:
                print(f"Warning: Missing columns in new data: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    new_data[col] = 0
            
            # Select only the required columns in the right order
            new_data = new_data[feature_columns]
        
        predictions = model.predict(new_data)
        print(f"Generated {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def main():
    """
    Main execution function that runs the complete ML pipeline.
    """
    # Start overall timer
    pipeline_start_time = time.time()
    start_timestamp = datetime.now()
    
    print("=" * 60)
    print("COAL PRODUCTION PREDICTION ML PIPELINE")
    print("=" * 60)
    print(f"Pipeline started at: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        step_start = time.time()
        df = load_and_prepare_data()
        step_start = print_step_time("Data Loading", step_start)
        
        # Step 2: Preprocess data
        X, y = preprocess_data(df)
        step_start = print_step_time("Data Preprocessing", step_start)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        step_start = print_step_time("Data Splitting", step_start)
        
        # Step 4: Compare models
        model_comparison = compare_models(X_train, y_train)
        step_start = print_step_time("Model Comparison", step_start)
        
        # Step 5: Get best model name
        best_model_name = model_comparison.iloc[0]['Model']
        print(f"\n✓ Best performing model: {best_model_name}")
        
        # Step 6: Tune best model
        best_model = tune_best_model(X_train, y_train, best_model_name)
        step_start = print_step_time("Model Hyperparameter Tuning", step_start)
        
        # Step 7: Train final model on full training set
        print("Training final model...")
        best_model.fit(X_train, y_train)
        step_start = print_step_time("Final Model Training", step_start)
        
        # Step 8: Evaluate model performance
        metrics = evaluate_model_performance(best_model, X_test, y_test, X_train, y_train)
        step_start = print_step_time("Model Evaluation", step_start)
        
        # Step 9: Feature importance analysis
        feature_importance = get_feature_importance(best_model, X_train.columns)
        step_start = print_step_time("Feature Importance Analysis", step_start)
        
        # Step 10: Save model to .sav format in saved_models folder
        model_path = save_model(best_model, 'saved_models/best_model.sav', include_metadata=True)
        step_start = print_step_time("Model Saving", step_start)
        
        # Save feature columns for future predictions
        feature_columns_path = 'saved_models/feature_columns.pkl'
        os.makedirs('saved_models', exist_ok=True)
        with open(feature_columns_path, 'wb') as f:
            pickle.dump(list(X_train.columns), f)
        print(f"✓ Feature columns saved as '{feature_columns_path}'")
        
        # Calculate total pipeline time
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        end_timestamp = datetime.now()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Started at:  {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished at: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time:  {format_duration(total_duration)}")
        print("=" * 60)
        print("RESULTS SUMMARY:")
        print("=" * 60)
        print(f"Best Model:     {best_model_name}")
        print(f"Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"Test RMSE:      {metrics['test_rmse']:.4f}")
        print(f"Test MAE:       {metrics['test_mae']:.4f}")
        print("=" * 60)
        print("SAVED FILES:")
        print("=" * 60)
        print(f"✓ Model:               {model_path}")
        print(f"✓ Model Metadata:      saved_models/best_model_metadata.pkl")
        print(f"✓ Feature Columns:     {feature_columns_path}")
        print(f"✓ Feature Importance:  feature_importance.csv")
        print(f"✓ Evaluation Plot:     model_evaluation.png")
        print(f"✓ Feature Plot:        feature_importance.png")
        print("=" * 60)
        print("The model is ready for deployment!")
        print("=" * 60)
        
        return best_model, feature_importance, model_comparison
        
    except Exception as e:
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        end_timestamp = datetime.now()
        
        print("\n" + "=" * 60)
        print("PIPELINE FAILED!")
        print("=" * 60)
        print(f"Started at:  {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Failed at:   {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time:  {format_duration(total_duration)}")
        print(f"Error: {e}")
        print("=" * 60)
        
        import traceback
        traceback.print_exc()
        raise

def demo_model_loading():
    """
    Demonstrate how to load and use the saved model.
    """
    print("\n" + "=" * 50)
    print("DEMO: Loading and Using Saved Model")
    print("=" * 50)
    
    try:
        # Load the saved model
        model = load_saved_model('saved_models/best_model.sav')
        
        # Load feature columns
        with open('saved_models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        print(f"Model loaded successfully!")
        print(f"Expected features: {len(feature_columns)}")
        print(f"Sample features: {feature_columns[:5]}")
        
        print("\nTo use this model for new predictions:")
        print("1. Load your new data")
        print("2. Preprocess it the same way as training data")
        print("3. Ensure it has the same feature columns")
        print("4. Use model.predict(new_data)")
        
    except Exception as e:
        print(f"Error in demo: {e}")

if __name__ == "__main__":
    # Run the complete ML pipeline
    try:
        model, importance, comparison = main()
        
        # Optional: Demo model loading
        demo_model_loading()
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        exit(1)
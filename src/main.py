import os
from src.data.preprocessing import load_data, preprocess_data
from src.visualization.plots import (
    plot_marital_status_vs_attrition,
    plot_department_vs_attrition,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_gender_vs_attrition,
    plot_age_vs_attrition
)
from src.models.train import (
    prepare_train_test_data,
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    get_feature_importance
)

def main():
    # Load and preprocess data
    data_path = os.path.join('src', 'data', 'raw', 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df = load_data(data_path)
    
    # Print initial data information
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    
    # Create visualizations before preprocessing
    plot_marital_status_vs_attrition(df)
    plot_department_vs_attrition(df)
    
    # Preprocess data
    df_processed, scaler = preprocess_data(df)
    
    # Create visualizations after preprocessing
    plot_correlation_heatmap(df_processed)
    
    # Prepare training and testing data
    X_train, X_test, Y_train, Y_test = prepare_train_test_data(df_processed)
    
    # Train and evaluate models
    print("\nTraining Logistic Regression Model...")
    lr_model = train_logistic_regression(X_train, Y_train)
    evaluate_model(lr_model, X_test, Y_test, "Logistic Regression")
    
    print("\nTraining Random Forest Model...")
    rf_model = train_random_forest(X_train, Y_train)
    evaluate_model(rf_model, X_test, Y_test, "Random Forest")
    
    # Feature importance analysis
    importances_df = get_feature_importance(rf_model, X_train.columns)
    plot_feature_importance(importances_df)
    
    # Additional visualizations
    plot_gender_vs_attrition(df)
    plot_age_vs_attrition(df)

if __name__ == "__main__":
    main() 
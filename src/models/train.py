import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """Prepare training and testing datasets."""
    X = df.drop('Attrition', axis=1)
    Y = df['Attrition']
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, Y_train):
    """Train logistic regression model."""
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, Y_train)
    return model

def train_random_forest(X_train, Y_train):
    """Train random forest model."""
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test, model_name):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    
    print(f'{model_name} Accuracy:', accuracy_score(Y_test, predictions))
    print('\nClassification Report:')
    print(classification_report(Y_test, predictions))
    print(f'{model_name} ROC-AUC:', roc_auc_score(Y_test, predictions))
    
    return predictions

def get_feature_importance(model, feature_names):
    """Get feature importance for Random Forest model."""
    importances = model.feature_importances_
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    return importances_df.sort_values(by='Importance', ascending=False) 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_marital_status_vs_attrition(df):
    """Plot marital status vs attrition distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='MaritalStatus', hue='Attrition', palette='Set2')
    plt.title('Marital Status vs. Attrition', fontsize=16)
    plt.xlabel('Marital Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition', labels=['No', 'Yes'])
    plt.show()

def plot_department_vs_attrition(df):
    """Plot department vs attrition distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Department', hue='Attrition', palette='husl')
    plt.title('Department vs. Attrition', fontsize=16)
    plt.xlabel('Department', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition', labels=['No', 'Yes'])
    plt.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of the dataset."""
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
    plt.title('Correlation HeatMap')
    plt.show()

def plot_feature_importance(importances_df):
    """Plot top 5 feature importance."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=importances_df.head(5),
        x='Importance',
        y='Feature',
        hue='Feature',
        legend=False
    )
    plt.title('Top 5 Features in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_gender_vs_attrition(df):
    """Plot gender vs attrition distribution."""
    if 'Gender' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Gender', hue='Attrition', palette='Set1')
        plt.title('Gender vs. Attrition', fontsize=16)
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Attrition', labels=['No', 'Yes'])
        plt.show()
    elif 'Gender_Male' in df.columns:
        df['Gender'] = df['Gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Gender', hue='Attrition', palette='Set1')
        plt.title('Gender vs. Attrition', fontsize=16)
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Attrition', labels=['No', 'Yes'])
        plt.show()

def plot_age_vs_attrition(df):
    """Plot age vs attrition distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Attrition', multiple='stack', palette='viridis', bins=20)
    plt.title('Age vs. Attrition', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition', labels=['No', 'Yes'])
    plt.show() 
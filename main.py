import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , roc_auc_score

# Loading the Dataset
df = pd.read_csv('src/data/raw/data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(df.head())
print("Columns in DataFrame:", df.columns)
print("Unique values in 'MaritalStatus':", df['MaritalStatus'].unique())
print("Value counts in 'MaritalStatus':")
print(df['MaritalStatus'].value_counts())
print(df[df['MaritalStatus'] == 'Divorced'].head())
print(df['MaritalStatus'].value_counts())
print(df.groupby(['MaritalStatus', 'Attrition']).size())

# print(df.columns())
print(df.info())
print(df.describe())
df = df.dropna()  # Dropping the rows with missing value
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Marital Status vs Attrition
# Write the code before on-hot encoding because if used after the Marital data name is replaced where graph is not visible
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='MaritalStatus', hue='Attrition', palette='Set2')
plt.title('Marital Status vs. Attrition', fontsize=16)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.savefig('marital_status_vs_attrition.png')
plt.close()

# Department vs. Attrition
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Department', hue='Attrition', palette='husl')
plt.title('Department vs. Attrition', fontsize=16)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.savefig('marital_status_vs_attrition.png')
plt.close()

df = pd.get_dummies(df, drop_first=True)  # Using One-hot encoding technique for data preprocessing
# converting non-numerical to numerical format

# Normalizing it for better performance
scaler = StandardScaler()
numerical_cols = ['MonthlyIncome', 'DistanceFromHome', 'Age']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Converting Result into Indian Metric
df['MonthlyIncome'] = df['MonthlyIncome'] * 85.80  # I am the current rate of 1USD of 85.80 rupees
df['DistanceFromHome'] = df['DistanceFromHome'].apply(lambda x: x * 1.5)  # I am Converting US miles into Kilometer

# For Visualizing Attrition Distribution with Exploratory Data Analysis (EDA)
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.savefig('attrition_distribution.png')
plt.close()
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation HeatMap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Making Test and Train set for the Model
X = df.drop('Attrition', axis=1)
Y = df['Attrition']
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using Machine Learning Models
# Training multiple models to compare performance

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
# Increasing rhe iteration to the Logistic Regression
lr_model.fit(X_train, Y_train)
lr_prediction = lr_model.predict(X_test)


# accuracy_score and classification_report
print('Logistical Regression Accuracy', accuracy_score(Y_test, lr_prediction))
print(classification_report(Y_test, lr_prediction))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)
rf_prediction = rf_model.predict(X_test)

# accuracy_score and classification_report
print('Random Forest Accuracy', accuracy_score(Y_test, rf_prediction))
print(classification_report(Y_test, rf_prediction))

# Using roc_auc_score to Evaluate the Model
print('Logistical Regression ROC-AUC', roc_auc_score(Y_test, lr_prediction))
print('Random Forest ROC-AUC', roc_auc_score(Y_test, rf_prediction))

# Feature  Importance Visualisation for Random Forest Model
# This helps to understand which features contribute to the Predictions
importances = rf_model.feature_importances_  # Extracting the feature importance score
features = X.columns  # Getting the feature names
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})  # Combining it into DataFrame
importances_df = importances_df.sort_values(by='Importance', ascending=False)  # Sorting by Importance

# Plotting the top 5 important Features
# colors = ['red', 'blue', 'green', 'orange', 'purple']
colors = plt.cm.viridis(np.linspace(0, 1, len(importances_df.head(5))))
plt.figure(figsize=(8, 5))  # Setting the size of the figures
sns.barplot(x='Importance', y='Feature', data=importances_df.head(5), palette=colors)
# Making a bar plot of top 5 features
plt.title('Top 5 Features in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance.png')
plt.close()

# Gender vs. Attrition
if 'Gender' in df.columns:  # If the Gender column exists in the original form
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Gender', hue='Attrition', palette=colors)
    plt.title('Gender vs. Attrition', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition', labels=['No', 'Yes'])
    plt.show()
elif 'Gender_Male' in df.columns:  # If Gender is one-hot encoded
    df['Gender'] = df['Gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Gender', hue='Attrition', palette='Set1')
    plt.title('Gender vs. Attrition', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition', labels=['No', 'Yes'])
    plt.show()
else:
    print("Gender column not found in the dataset!")


# Age vs. Attrition
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Attrition', multiple='stack', palette='viridis', bins=20)
plt.title('Age vs. Attrition', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.savefig('marital_status_vs_attrition.png')
plt.close()













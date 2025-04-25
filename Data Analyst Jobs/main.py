import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

data = pd.read_csv("data_analyst_jobs.csv")

print(data.head())
print(data.info())

# Check for duplicates
print(f"Duplicate rows: {data.duplicated().sum()}")

# General statistics
print(data.describe(include='all'))

# Value counts for categorical columns
for col in ['Job Title', 'Type of ownership', 'Industry','Sector']:
    print(data[col].value_counts().head())

# Salary distribution
plt.figure(figsize=(10, 6))

sns.histplot(data['Salary Estimate'], kde=True, bins=20)
plt.title("Salary Estimate Distribution")
plt.xlabel("Salary")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Industry', y='Rating', data=data)
plt.xticks(rotation=90)
plt.title("Company Ratings by Industry")
plt.show()

# Check missing values
print(data.isnull().sum())

# Fill missing numerical values
data['Rating'].fillna(data['Rating'].median(), inplace=True)
# Drop columns with > 30% missing data
threshold = len(data) * 0.3
data = data.dropna(thresh=threshold, axis=1)

# Forward-fill categorical values
categorical_cols = ['Company Name', 'Industry', 'Sector', 'Type of ownership']
data[categorical_cols] = data[categorical_cols].fillna(method='ffill')

# Extract minimum salary
data['Min Salary'] = data['Salary Estimate'].str.extract(r'(\d+)').astype(float)
# Extract maximum salary
data['Max Salary'] = data['Salary Estimate'].str.extract(r'-\s*(\d+)').astype(float)
# Compute average salary
data['Avg Salary'] = (data['Min Salary'] + data['Max Salary'])/ 2
# Drop old salary column
data.drop('Salary Estimate', axis=1, inplace=True)
# Extract keywords from Job Description
data['Python'] = data['Job Description'].str.contains('Python',
case=False, na=False).astype(int)
data['Excel'] = data['Job Description'].str.contains('Excel',
case=False, na=False).astype(int)
# Create a tech skills score
data['Tech_Skills'] = data['Python'] + data['Excel']
# Extract city and state from location
data['City'] = data['Location'].str.split(',', expand=True)[0]
data['State'] = data['Location'].str.split(',', expand=True)[1]

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Define features and target
features = ['Rating', 'Tech_Skills', 'Size', 'Founded']
X = data[features]
y = data['Avg Salary']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
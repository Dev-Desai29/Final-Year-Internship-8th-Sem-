import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("Data_Science_Job_Salaries.csv")

# Quick overview
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Fill missing values (example strategies)
data['salary_in_usd'].fillna(data['salary_in_usd'].median(), inplace=True) # Replace with median
data['company_size'].fillna('Unknown', inplace=True) # Replace missing categories with 'Unknown'

# Drop rows with critical missing data
data.dropna(subset=['job_title', 'experience_level'], inplace=True)

# Verify no missing values remain
print(data.isnull().sum())

# Standardize text case for categorical columns
data['job_title'] = data['job_title'].str.lower()
data['company_size'] = data['company_size'].str.capitalize()

# Verify unique values
print(data['job_title'].unique())
print(data['company_size'].unique())

# Encode categorical variables
data['experience_level'] = data['experience_level'].map({'EN':0, 'MI': 1, 'SE': 2, 'EX': 3})
data['employment_type'] = data['employment_type'].map({'PT': 0,'FT': 1, 'CT': 2, 'FL': 3})

# Summary statistics
print(data.describe())

plt.figure(figsize=(10, 6))
sns.histplot(data['salary_in_usd'], bins=30, kde=True, color='blue')
plt.title('Salary Distribution (USD)')
plt.xlabel('Salary in USD')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Boxplot for salaries by experience level
plt.figure(figsize=(12, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=data)
plt.title('Salary by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary in USD')
plt.show()

# Remote ratio vs salary
plt.figure(figsize=(12, 6))
sns.barplot(x='remote_ratio', y='salary_in_usd', data=data)
plt.title('Salary by Remote Ratio')
plt.xlabel('Remote Ratio')
plt.ylabel('Salary in USD')
plt.show()

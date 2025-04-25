import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('daily_transactions.csv')
# Display the first few rows of the dataset
df.head()


# Check for missing values
df.isnull().sum()
# Fill or drop missing values
df['Category'].fillna('Unknown', inplace=True)
df.dropna(subset=['Date', 'Transaction_ID', 'Amount'], inplace=True)
# Convert data types
df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = df['Amount'].astype(float)
# Remove duplicates
df.drop_duplicates(inplace=True)
# Verify data types
df.dtypes

# Summary statistics
df.describe()
# Distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Transaction counts by category
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Category', order=df['Category'].value_counts().index)
plt.title('Transaction Counts by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Transaction counts by type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Type')
plt.title('Transaction Counts by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# Resample data to monthly frequency
monthly_data = df.resample('M', on='Date').sum()
plt.figure(figsize=(14, 7))
plt.plot(monthly_data.index, monthly_data['Amount'], marker='o')
plt.title('Monthly Transaction Amounts')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.grid(True)
plt.show()

# Daily trends
daily_data = df.groupby(df['Date'].dt.date).sum()
plt.figure(figsize=(14, 7))
plt.plot(daily_data.index, daily_data['Amount'], marker='o')
plt.title('Daily Transaction Amounts')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.grid(True)
plt.show()

# Create a pivot table for correlation analysis
pivot_table = df.pivot_table(index='Date', columns='Category', values='Amount', aggfunc='sum', fill_value=0)
# Calculate correlation matrix
correlation_matrix = pivot_table.corr()# Create a pivot table for correlation analysis
pivot_table = df.pivot_table(index='Date', columns='Category', values='Amount', aggfunc='sum', fill_value=0)
# Calculate correlation matrix
correlation_matrix = pivot_table.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Transaction Categories')
plt.show()


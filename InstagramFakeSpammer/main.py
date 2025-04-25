import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('train.csv')
data.head()
# Check for missing values
data.isnull().sum()

# Basic statistics of each column
data.describe()

# Plotting the distribution of fake and genuine accounts
sns.countplot(x='fake', data=data)
plt.title("Distribution of Fake vs Genuine Accounts")
plt.show()

# Correlation matrix
correlation = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

sns.barplot(x='fake', y='profile_pic', data=data)
plt.title("Profile Picture Presence in Fake vs Genuine Accounts")
plt.show()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('fake', axis=1))
scaled_data = pd.DataFrame(scaled_features,
columns=data.columns[:-1])
scaled_data['fake'] = data['fake']

# Split data into training and test sets

X = scaled_data.drop('fake', axis=1)
y = scaled_data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
sns.barplot(y=X.columns[indices], x=importances[indices], palette='viridis')
plt.show()
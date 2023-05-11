import pandas as pd

# Read data from a CSV file
data = pd.read_csv('data.csv')

# Handling Missing Values
# Drop rows with missing values
cleaned_data = data.dropna()

# Fill missing values with mean
data['column_name'].fillna(data['column_name'].mean(), inplace=True)

# Impute missing categorical values with mode
data['column_name'].fillna(data['column_name'].mode()[0], inplace=True)

# Handling Outliers
# Remove outliers based on z-score
z_scores = (data['column_name'] - data['column_name'].mean()) / data['column_name'].std()
outlier_threshold = 3
filtered_data = data[z_scores.abs() < outlier_threshold]

# Handling Duplicates
# Drop duplicate rows
data.drop_duplicates(inplace=True)

# Handling Inconsistent Values
# Replace inconsistent values
data['column_name'].replace('inconsistent_value', 'correct_value', inplace=True)

# Text Cleaning
# Remove leading/trailing whitespaces
data['text_column'] = data['text_column'].str.strip()

# Convert text to lowercase
data['text_column'] = data['text_column'].str.lower()

# Data Type Conversion
# Convert column to datetime
data['date_column'] = pd.to_datetime(data['date_column'])

# Convert column to numeric
data['numeric_column'] = pd.to_numeric(data['numeric_column'])

# Handling Categorical Data
# Convert categorical column to one-hot encoded
one_hot_encoded = pd.get_dummies(data['categorical_column'])

# Merge one-hot encoded columns with the original data
data = pd.concat([data, one_hot_encoded], axis=1)

# Removing Unused Columns
# Drop unnecessary columns
columns_to_drop = ['column1', 'column2']
data.drop(columns_to_drop, axis=1, inplace=True)

# Save cleaned data to a new CSV file
data.to_csv('cleaned_data.csv', index=False)



import pandas as pd

df = pd.read_csv('/content/Amazon_popular_books_dataset (1).csv')

df.head()

list_of_columns = df.columns

list_of_columns

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("Categorical Columns:", list(categorical_columns))

columns_to_encode = [
    'asin', 'ISBN10', 'availability', 'brand', 'currency', 'date_first_available',
    'delivery', 'description', 'domain', 'features', 'format', 'image_url',
    'manufacturer', 'model_number', 'product_dimensions', 'seller_id',
    'seller_name', 'timestamp', 'title', 'url', 'categories', 'buybox_seller', 'colors'
]

from sklearn.preprocessing import LabelEncoder
import pandas as pd

le = LabelEncoder()
columns_to_encode = [
    'asin', 'ISBN10', 'availability', 'brand', 'currency', 'date_first_available',
    'delivery', 'description', 'domain', 'features', 'format', 'image_url',
    'manufacturer', 'model_number', 'product_dimensions', 'seller_id',
    'seller_name', 'timestamp', 'title', 'url', 'categories', 'buybox_seller', 'colors','rating'
]


for col in columns_to_encode:

    if df[col].dtype == 'object':

        df[col] = le.fit_transform(df[col])
    else:

        pass

import pandas as pd
import re

def convert_weights_tonumeric(df, column_name):

    if column_name in df:

        numbers = []
        for s in df[column_name]:
            match = re.search(r"[\d.]+", str(s))
            if match:
                numbers.append(float(match.group()))
            else:
                numbers.append(None)  
        df[column_name] = numbers  
    else:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")
    return df



print(f"Number of rows in the dataset: {len(df)}")


df = convert_weights_tonumeric(df, 'item_weight')



# Set display option and print to confirm all rows
# pd.set_option('display.max_rows', None)  # Show all rows
# print(df[['item_weight']])

# Import necessary libraries
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import re
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Step 1: Load the dataset
df = pd.read_csv('/content/Amazon_popular_books_dataset (1).csv')

# Step 2: Remove columns with all missing values
missing_columns = df.columns[df.isnull().all()]
df = df.drop(columns=missing_columns)
print(f"Removed columns with all missing values: {list(missing_columns)}")

# Step 3: Handle missing values using SimpleImputer (filling missing values with the mean)
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Impute only numeric columns
data_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Fill missing values for non-numeric columns with "Unknown"
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")

# Step 4: Simplify JSON-like columns
def process_json_column(column):
    """Simplify JSON-like data by extracting key fields."""
    simplified_values = []
    for value in column:
        try:
            json_data = json.loads(value.replace("'", '"'))  # Parse JSON data
            if isinstance(json_data, list) and len(json_data) > 0:
                simplified_values.append(str(json_data[0]))  # Take the first item
            else:
                simplified_values.append(str(json_data))
        except (json.JSONDecodeError, TypeError):
            simplified_values.append(str(value))  # Fallback to string representation
    return simplified_values

json_columns = ['categories', 'best_sellers_rank']
for col in json_columns:
    if col in df.columns:
        df[col] = process_json_column(df[col])

# Step 5: Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column is of object type
        df[col] = le.fit_transform(df[col].astype(str))

# Step 6: Convert weights and ratings to numeric values (if applicable)
def convert_to_numeric(df, column_name):
    if column_name in df:
        numbers = []
        for s in df[column_name]:
            match = re.search(r"[\d.]+", str(s))  # Extract numeric values
            if match:
                numbers.append(float(match.group()))  # Extract and convert to float
            else:
                numbers.append(None)  # Append None if no numeric value is found
        df[column_name] = numbers  # Update the column with numeric values
    return df

# Convert the relevant columns to numeric
df = convert_to_numeric(df, 'item_weight')  # Convert item_weight if it exists
df = convert_to_numeric(df, 'rating')       # Convert rating if it exists

# Step 7: Impute missing numeric values after conversion
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Step 8: Compute the correlation matrix using only numeric columns
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Optional: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Import necessary libraries
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import re
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Step 1: Load the dataset
df = pd.read_csv('/content/Amazon_popular_books_dataset (1).csv')

# Step 2: Remove columns with all missing values
missing_columns = df.columns[df.isnull().all()]
df = df.drop(columns=missing_columns)
print(f"Removed columns with all missing values: {list(missing_columns)}")

# Step 3: Handle missing values using SimpleImputer (filling missing values with the mean)
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Impute only numeric columns
data_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Fill missing values for non-numeric columns with "Unknown"
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")

# Step 4: Simplify JSON-like columns
def process_json_column(column):
    """Simplify JSON-like data by extracting key fields."""
    simplified_values = []
    for value in column:
        try:
            json_data = json.loads(value.replace("'", '"'))  # Parse JSON data
            if isinstance(json_data, list) and len(json_data) > 0:
                simplified_values.append(str(json_data[0]))  # Take the first item
            else:
                simplified_values.append(str(json_data))
        except (json.JSONDecodeError, TypeError):
            simplified_values.append(str(value))  # Fallback to string representation
    return simplified_values

json_columns = ['categories', 'best_sellers_rank']
for col in json_columns:
    if col in df.columns:
        df[col] = process_json_column(df[col])

# Step 5: Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column is of object type
        df[col] = le.fit_transform(df[col].astype(str))

# Step 6: Convert weights and ratings to numeric values (if applicable)
def convert_to_numeric(df, column_name):
    if column_name in df:
        numbers = []
        for s in df[column_name]:
            match = re.search(r"[\d.]+", str(s))  # Extract numeric values
            if match:
                numbers.append(float(match.group()))  # Extract and convert to float
            else:
                numbers.append(None)  # Append None if no numeric value is found
        df[column_name] = numbers  # Update the column with numeric values
    return df

# Convert the relevant columns to numeric
df = convert_to_numeric(df, 'item_weight')  # Convert item_weight if it exists
df = convert_to_numeric(df, 'rating')       # Convert rating if it exists

# Step 7: Impute missing numeric values after conversion
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Step 8: Compute the correlation matrix using all columns (including 'final_price' and other numeric columns)
correlation_matrix = df.corr()

# Step 9: Extract the correlation of 'final_price' with all other columns
if 'final_price' in df.columns:
    final_price_corr = correlation_matrix['final_price']
    print("Correlation of Final Price with All Columns:")
    print(final_price_corr)
else:
    print("Column 'final_price' not found in the dataset.")

# Optional: Visualize the correlation of 'final_price' with all features using a bar plot
plt.figure(figsize=(10, 6))
final_price_corr.drop('final_price').sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title('Correlation of Final Price with Other Features')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.show()

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
  df = df[['final_price','initial_price', 'discount', 'best_sellers_rank', 'seller_name', 'delivery','seller_id']]
  return df

preprocessed_df = preprocess_data(df)

preprocessed_df.head()

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Assuming 'preprocessed_df' is your DataFrame
X = preprocessed_df.drop('final_price', axis=1)
y = preprocessed_df['final_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest Regression': RandomForestRegressor(),
    'XGBoost Regression': xgb.XGBRegressor(n_estimators=1000, random_state=42)
}

# Dictionary to store results
model_errors = {}
model_r2_scores = {}

# Train each model, predict and compute error and R2
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    error = mean_squared_error(y_test, y_pred)
    model_errors[model_name] = error

    # Calculate R-squared (R²)
    r2 = r2_score(y_test, y_pred)
    model_r2_scores[model_name] = r2

# Plotting MSE values
plt.figure(figsize=(10, 6))
plt.barh(list(model_errors.keys()), list(model_errors.values()), color='skyblue')
plt.xlabel('Mean Squared Error (MSE)')
plt.title('Model Comparison Based on MSE')
plt.tight_layout()
plt.show()

# Plotting R² values
plt.figure(figsize=(10, 6))
plt.barh(list(model_r2_scores.keys()), list(model_r2_scores.values()), color='lightgreen')
plt.xlabel('R² (R-squared)')
plt.title('Model Comparison Based on R²')
plt.tight_layout()
plt.show()

# Find the best model based on MSE and R²
best_model_mse = min(model_errors, key=model_errors.get)
best_model_r2 = max(model_r2_scores, key=model_r2_scores.get)

print(f'Best Model based on MSE: {best_model_mse} with MSE: {model_errors[best_model_mse]}')
print(f'Best Model based on R²: {best_model_r2} with R²: {model_r2_scores[best_model_r2]}')

# Choose the best model (can be based on MSE or R², here choosing based on R²)
best_model_name = best_model_r2  # or choose best_model_mse if based on MSE
best_model = models[best_model_name]

# Save the best model to a .pkl file
model_filename = f'{best_model_name}_model.pkl'
joblib.dump(best_model, model_filename)

print(f'Best model saved as {model_filename}')

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_test (actual) and y_pred (predicted) are already defined
y_pred = best_model.predict(X_test)  # Or model.predict(X_test) if using another model

# 1. Scatter Plot (Predicted vs Actual)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel('Actual Final Price')
plt.ylabel('Predicted Final Price')
plt.title('Actual vs Predicted Final Price')
plt.legend()
plt.tight_layout()
plt.show()



# 3. Distribution Plot for Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual Final Price', color='blue', fill=True, alpha=0.3)
sns.kdeplot(y_pred, label='Predicted Final Price', color='orange', fill=True, alpha=0.3)
plt.xlabel('Final Price')
plt.ylabel('Density')
plt.title('Actual vs Predicted Final Price Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Actual vs Predicted Line Plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Final Price', color='blue', linestyle='-', linewidth=2)
plt.plot(y_pred, label='Predicted Final Price', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Final Price')
plt.title('Actual vs Predicted Final Price (Line Plot)')
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming y_test (actual) and y_pred (predicted) are already defined
# Combine actual and predicted data for comparison
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# 1. Line Plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Actual'].values, label='Actual Final Price', color='blue', linewidth=2)
plt.plot(comparison_df['Predicted'].values, label='Predicted Final Price', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Final Price')
plt.title('Actual vs Predicted Final Price (Line Plot)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Plot: Comparison for First 20 Values
sample_size = 20  # Display first 20 samples
comparison_sample = comparison_df.head(sample_size)

plt.figure(figsize=(12, 6))
plt.bar(range(sample_size), comparison_sample['Actual'], width=0.4, label='Actual Final Price', color='blue')
plt.bar([i + 0.4 for i in range(sample_size)], comparison_sample['Predicted'], width=0.4, label='Predicted Final Price', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Final Price')
plt.title('Actual vs Predicted Final Price (Bar Plot)')
plt.xticks([i + 0.2 for i in range(sample_size)], range(sample_size))
plt.legend()
plt.tight_layout()
plt.show()

# 3. KDE Distribution Plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.kdeplot(comparison_df['Actual'], label='Actual Final Price', color='blue', fill=True, alpha=0.4)
sns.kdeplot(comparison_df['Predicted'], label='Predicted Final Price', color='orange', fill=True, alpha=0.4)
plt.xlabel('Final Price')
plt.ylabel('Density')
plt.title('Actual vs Predicted Final Price Distribution')
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Realistic hardcoded data (replace with your actual data structure and values)
data = {
    'final_price': [19.99, 25.50, 14.99, 22.75, 18.30],  # Actual final price of books
    'initial_price': [29.99, 35.00, 19.99, 27.00, 23.00],  # Initial price before discount
    'discount': [0.33, 0.27, 0.25, 0.16, 0.21],  # Discount applied (as a percentage)
    'best_sellers_rank': ['Rank10', 'Rank5', 'Rank20', 'Rank1', 'Rank8'],  # Book best sellers rank
    'seller_name': ['Amazon', 'Barnes & Noble', 'Books-A-Million', 'Amazon', 'Barnes & Noble'],  # Seller name
    'delivery': ['Free', 'Paid', 'Free', 'Paid', 'Free'],  # Delivery type (e.g., free or paid shipping)
    'seller_id': ['A1', 'B2', 'C3', 'A1', 'B2']  # Seller ID
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocess the data function
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['final_price', 'initial_price', 'discount', 'best_sellers_rank', 'seller_name', 'delivery', 'seller_id']]

    # Convert categorical columns to numeric using LabelEncoder
    label_encoder = LabelEncoder()

    # Encoding categorical columns ('seller_name', 'delivery', 'seller_id')
    df['seller_name'] = label_encoder.fit_transform(df['seller_name'])
    df['delivery'] = label_encoder.fit_transform(df['delivery'])
    df['seller_id'] = label_encoder.fit_transform(df['seller_id'])

    # Convert 'best_sellers_rank' to numeric
    df['best_sellers_rank'] = pd.to_numeric(df['best_sellers_rank'].str.extract('(\d+)')[0], errors='coerce')
    df['best_sellers_rank'] = df['best_sellers_rank'].fillna(df['best_sellers_rank'].median())  # Fill NaN with median

    return df

# Preprocess the dataframe
preprocessed_df = preprocess_data(df)

# Load the saved model (make sure the model is already trained and saved)
loaded_model = joblib.load('XGBoost Regression_model.pkl')  # Replace with the actual filename of your model

# Make predictions
y_pred = loaded_model.predict(preprocessed_df.drop('final_price', axis=1))

# Add the predicted prices to the dataframe
preprocessed_df['Predicted_Final_Price'] = y_pred

# Display the actual vs predicted prices in a table
print(preprocessed_df[['final_price', 'Predicted_Final_Price']])


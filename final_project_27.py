import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\\BrentOilPrices.csv')

# Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'],)

# Sort by date
data = data.sort_values('Date')

# Create lagged features
data['Price_Lag1'] = data['Price'].shift(1)
data['Price_Lag2'] = data['Price'].shift(2)
data['Price_Lag3'] = data['Price'].shift(3)

# Drop rows with NaN values
data = data.dropna()

# Define features and target
X = data[['Price_Lag1', 'Price_Lag2', 'Price_Lag3']]
y = (data['Price'] > data['Price'].shift(-1)).astype(int)  # Binary classification: 1 if price increases, 0 otherwise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Classifier
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# MLP Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return accuracy, precision, recall

svm_metrics = evaluate_model(y_test, y_pred_svm, 'SVM')
mlp_metrics = evaluate_model(y_test, y_pred_mlp, 'MLP')
dt_metrics = evaluate_model(y_test, y_pred_dt, 'Decision Tree')

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Model': ['SVM', 'MLP', 'Decision Tree'],
    'Accuracy': [svm_metrics[0], mlp_metrics[0], dt_metrics[0]],
    'Precision': [svm_metrics[1], mlp_metrics[1], dt_metrics[1]],
    'Recall': [svm_metrics[2], mlp_metrics[2], dt_metrics[2]]
})

# Plot the metrics
metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison: Accuracy, Precision, and Recall')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

# Print the metrics
print(metrics_df)

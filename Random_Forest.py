import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Load the data from the CSV file
data = pd.read_csv("C:/Users/Sheeraz/Downloads/ffinalconcatenated.csv")

# Separate the features and the labels
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the Random Forest model with tuned parameters
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

for i in range(1, model.n_estimators + 1):
    model.set_params(n_estimators=i)
    model.fit(X_train, y_train)
    print(f"Training progress: {i}/{model.n_estimators}")

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:")
print(confusion_mat)

# Plotting the confusion matrix
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Displaying the values in the confusion matrix
for i in range(len(confusion_mat)):
    for j in range(len(confusion_mat)):
        plt.text(j, i, str(confusion_mat[i][j]), ha='center', va='center', color='white')

plt.show()

# Increasing complexity and performance
model_complex = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)

for i in range(1, model_complex.n_estimators + 1):
    model_complex.set_params(n_estimators=i)
    model_complex.fit(X_train, y_train)
    print(f"Complex model training progress: {i}/{model_complex.n_estimators}")

y_pred_complex = model_complex.predict(X_test)

f1_complex = f1_score(y_test, y_pred_complex, average='weighted')
precision_complex = precision_score(y_test, y_pred_complex, average='weighted')
recall_complex = recall_score(y_test, y_pred_complex, average='weighted')
confusion_mat_complex = confusion_matrix(y_test, y_pred_complex)

print("F1 Score (Complex Model):", f1_complex)
print("Precision (Complex Model):", precision_complex)
print("Recall (Complex Model):", recall_complex)
print("Confusion Matrix (Complex Model):")
print(confusion_mat_complex)


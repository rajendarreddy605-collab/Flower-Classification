# Flower-Classification
The Iris Flower Classification project is a supervised machine learning application that predicts the species of an Iris flower based on its physical measurements. The model classifies flowers into one of three categories:
## Source Code

```python
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Iris.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Id Column
df = df.drop("Id", axis=1)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Species Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="Species", data=df)
plt.title("Species Distribution")
plt.show()

# Pair Plot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop("Species", axis=1).corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Convert Species Labels into Numeric Values
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Sample Prediction
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample_flower)

species_name = encoder.inverse_transform(prediction)

print("\nPredicted Species:")
print(species_name[0])
```

## Requirements

Create a file named `requirements.txt`

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Sample Output

```text
Accuracy Score:
100.0 %

Predicted Species:
Iris-setosa
```

## Screenshots to Upload

1. Dataset Preview
2. Dataset Information
3. Species Distribution Graph
4. Pair Plot Visualization
5. Correlation Heatmap
6. Confusion Matrix
7. Accuracy Score Output
8. Species Prediction Output

```
```

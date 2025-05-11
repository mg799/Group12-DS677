"""
Thyroid Cancer Risk Prediction using a Deep Feedforward Neural Network
This script trains a neural network to predict thyroid cancer risk based on a dataset."""

#########################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

#Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#Load and preprocess data (same as provided code)
print("Loading the dataset...")
data = pd.read_csv('thyroid_cancer_risk_data.csv')

##Map Diagnosis 
data['THYROID_CANCER_RISK'] = data['Diagnosis'].apply(lambda x: 0 if x == 'Benign' else 1)
data = data.drop(['Patient_ID', 'Diagnosis', 'Thyroid_Cancer_Risk'], axis=1)

##Define numerical and categorical columns
numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
categorical_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 'Radiation_Exposure', 
                   'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

##Fill missing values
for column in numerical_cols:
    data[column].fillna(data[column].median(), inplace=True)
for column in categorical_cols:
    data[column].fillna(data[column].mode()[0], inplace=True)

##Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

##Feature engineering (same interaction terms)
data['TSH_Nodule'] = data['TSH_Level'] * data['Nodule_Size']
data['Age_Family'] = data['Age'] * data['Family_History']
data['TSH_Radiation'] = data['TSH_Level'] * data['Radiation_Exposure']
data['Nodule_Family'] = data['Nodule_Size'] * data['Family_History']
data['T3_T4'] = data['T3_Level'] * data['T4_Level']
data['Age_TSH'] = data['Age'] * data['TSH_Level']
numerical_cols.extend(['TSH_Nodule', 'Age_Family', 'TSH_Radiation', 'Nodule_Family', 'T3_T4', 'Age_TSH'])

##Prepare features and target
X = data.drop('THYROID_CANCER_RISK', axis=1)
y = data['THYROID_CANCER_RISK']

##Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

##Split data
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Apply undersampling 
undersampler = RandomUnderSampler(sampling_strategy=0.333, random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train, y_train)

#Define deep feedforward neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train_balanced, y_train_balanced,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=1)

y_pred_prob = model.predict(X_test)
threshold = 0.5 
y_pred = (y_pred_prob > threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Set Performance (threshold = {threshold}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))

#Test multiple thresholds
thresholds = [0.4, 0.45, 0.5]
for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int)
    print(f"\nTest Set Performance (threshold = {threshold}):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

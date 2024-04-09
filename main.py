import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

#Functions for visualization
def analyze_distribution(data, column_name):
    column_distribution = data[column_name].value_counts()
    print(f"{column_name.capitalize()} Distribution:")
    print(column_distribution)
    plt.figure(figsize=(6, 4))
    plt.bar(column_distribution.index, column_distribution.values, color=['blue', 'pink'])
    plt.title(f'{column_name.capitalize()} Distribution')
    plt.xlabel(f'{column_name.capitalize()}')
    plt.ylabel('Count')
    plt.show()

def investigate_investment_behavior(data, column_name):
    column_distribution = data[column_name].value_counts()
    print(f"\n{column_name.capitalize()} Distribution:")
    print(column_distribution)
    plt.figure(figsize=(8, 6))
    plt.pie(column_distribution, labels=column_distribution.index, autopct='%1.1f%%', colors=['lightgreen', 'lightblue', 'orange'])
    plt.title(f'{column_name.capitalize()} Distribution')
    plt.show()

def visualize_feature_importance(features, importances):
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.show()

#Functions for Data Preprocessing
def preprocess_data(data):
    numerical_columns = ['Number of investors in family', 'Knowledge level about different investment product',
                     'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes']
    categorical_columns = ['Age', 'Role', 'Investment Influencer', 'Gender', 'Education', 'City',
                       'Reason for Investment', 'Household Income', 'Percentage of Investment',
                       'Investment Experience', 'Source of Awareness about Investment', 'Marital Status']

    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    #One Hot Encoding for Categorical Columns and Standard Scaling for Numerical Columns
    encoder = OneHotEncoder(drop='first')
    encoded_categorical = pd.DataFrame(encoder.fit_transform(data[categorical_columns]).toarray(),columns=encoder.get_feature_names_out(categorical_columns))
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(data[numerical_columns])
    scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical_columns)
    processed_data = pd.concat([encoded_categorical, scaled_numerical], axis=1)
    return processed_data

#Functions for Model Training and Evaluation
def train_model(X, y, rf_classifier):
    multi_output_rf_classifier = MultiOutputClassifier(rf_classifier)
    multi_output_rf_classifier.fit(X, y)
    return multi_output_rf_classifier

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    target_accuracies = [accuracy_score(y_test[:, i], predictions[:, i]) for i in range(y_test.shape[1])]
    target_classification_reports = []
    target_variable_names = ['Risk Level', 'Return Earned']
    for i, target_name in enumerate(target_variable_names):
        report = classification_report(y_test[:, i], predictions[:, i])
        target_classification_reports.append((target_name, report))
    print(f"Overall Accuracy: {np.mean(target_accuracies):.4f}")
    for target_name, report in target_classification_reports:
        print(f"\nClassification Report for {target_name}:\n{report}")

#Functions for encoding input for Prediction
def encode_input(input_data, category_values):
    encoded_columns = []
    for col, values in category_values.items():
        for val in values:
            encoded_columns.append(f"{col}_{val}")
    input_encoded = pd.DataFrame(columns=encoded_columns)
    for idx, row in input_data.iterrows():
        encoded_row = []
        for col, values in category_values.items():
            for val in values:
                if row[col] == val:
                    encoded_row.append(1)
                else:
                    encoded_row.append(0)
        input_encoded.loc[idx] = encoded_row
    return input_encoded

#Function for Prediction of new data
def get_prediction(model, X_train, label_encoder_risk, label_encoder_return, dropdowns, output_frame, categorical_columns, category_values):
    target_variable_names = ['Risk Level', 'Return Earned']
    user_inputs = {}
    for col in categorical_columns:
        user_inputs[col] = dropdowns[col].get()
    
    new_input = pd.DataFrame(user_inputs, index=[0])
    new_input_encoded = encode_input(new_input, category_values)
    new_input_encoded = new_input_encoded.reindex(columns=X_train.columns, fill_value=0)
    new_predictions = model.predict(new_input_encoded)
    decoded_predictions = np.column_stack((label_encoder_risk.inverse_transform(new_predictions[:, 0]), label_encoder_return.inverse_transform(new_predictions[:, 1])))

    for i, target_name in enumerate(target_variable_names):
        prediction_label = ttk.Label(output_frame, text=f"{target_name}: {decoded_predictions[0, i]}")
        prediction_label.grid(row=i, column=0, sticky="w")

#Loading data
data = pd.read_excel("sample_data.xlsx")

#Data Exploration
for column in ['Gender','Marital Status', 'Household Income']:
    analyze_distribution(data, column)

for column in ['Percentage of Investment', 'Investment Influencer', 'Risk Level']:
    investigate_investment_behavior(data, column)

#Data Preprocessing
processed_data = preprocess_data(data)
X = processed_data

#Encoding target variables as labels
label_encoder_risk = LabelEncoder()
label_encoder_return = LabelEncoder()
outcome_encoded_risk = label_encoder_risk.fit_transform(data['Risk Level'])
outcome_encoded_return = label_encoder_return.fit_transform(data['Return Earned'])
outcome_encoded = np.column_stack((outcome_encoded_risk, outcome_encoded_return))

#Feature Selection using Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, outcome_encoded)
feature_importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
top_features = sorted_indices[:10]
selected_features = X.columns[top_features]
print("Selected features : ",selected_features)
visualize_feature_importance(selected_features, feature_importances[top_features])

# Training and evaluating Model using the selected Features
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, outcome_encoded, test_size=0.2, random_state=42)
model = train_model(X_train, y_train, rf_classifier)
evaluate_model(model, X_test, y_test)

# Defining categories for dropdowns in UI
category_values = {}
for col in ['Age', 'Role', 'Investment Influencer', 'Gender', 'Education', 'City', 'Reason for Investment', 'Household Income', 'Percentage of Investment', 'Investment Experience', 'Source of Awareness about Investment', 'Marital Status']:
    category_values[col] = data[col].unique()

# Creating a Tkinter window
window = tk.Tk()
window.title("Predictor")

# Creating a frame for input widgets
input_frame = ttk.Frame(window)
input_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Creating dropdowns for each feature
dropdowns = {}
cols = data.drop(columns=['Risk Level', 'Return Earned', 'S. No.']).columns
for i, column in enumerate(cols):
    label = ttk.Label(input_frame, text=column)
    label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
    dropdown = ttk.Combobox(input_frame, values=data[column].unique().tolist(), state="readonly")
    dropdown.grid(row=i, column=1, padx=5, pady=5, sticky="we")
    dropdown.current(0)
    dropdowns[column] = dropdown

# Creating a frame for output labels
output_frame = ttk.Frame(window)
output_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Button to trigger the prediction
predict_button = ttk.Button(window, text="Get Prediction", command=lambda: get_prediction(model, X_train, label_encoder_risk, label_encoder_return, dropdowns, output_frame, ['Age', 'Role', 'Investment Influencer', 'Gender', 'Education', 'City', 'Reason for Investment', 'Household Income', 'Percentage of Investment', 'Investment Experience', 'Source of Awareness about Investment', 'Marital Status'], category_values))
predict_button.pack(padx=20, pady=10)
window.mainloop()


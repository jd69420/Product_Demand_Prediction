import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

path = r"cleaned.csv" 
df = pd.read_csv(path)

df = df.fillna(0)

#y = df['demand']
#X = df.drop(['demand','id'], axis=1)  # Exclude 'Lead Number'

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(df[['s_id', 'total', 'base', 'sold']], df['demand'], test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Accuracy: ", int(accuracy * 100), '%')

sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True,
            fmt='.2%', cmap='Blues').set(title='Confusion matrix: Logistic Regression')

tp = conf_matrix[0][0]
tn = conf_matrix[0][1]

print("\n True Positives: ", tp)
print("\n True Negatives: ",tn)

print("\nClassification Report: \n", class_report)
print("Confusion Matrix Loaded")


Z = pd.read_csv('x.csv') #test set 
print(Z)
feature_values = {}
for column in Z.columns:
    if column == "Unnamed: 0":
        feature_values[column] = 1
    elif column == "s_id":
        value = input(f"Enter the value for {column} (80/90 _ _): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "total":
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "base":
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "sold":
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  
        feature_values[column] = value


user_input = pd.DataFrame(feature_values, index=[0])

# Make predictions on the user input
user_pred = model.predict(user_input)

# Convert the predictions to a DataFrame or use as needed
#user_predictions_df = pd.DataFrame({'Converted': user_pred})
if user_pred == 0:
    print("\nProduct is of less Focus")
    print("MODEL SCORE: AVERAGE DEMAND\n")
else:
    print("\nProduct is of Higher Focus")
    print("MODEL SCORE: HIGH DEMAND\n")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

def train_and_save_model(data_path):
    """
    Loads data, trains a machine learning pipeline with a Random Forest Classifier,
    and saves the trained model to a file.

    This function addresses class imbalance using SMOTE and handles both
    numerical and categorical features.
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        print("Data loaded successfully.")

        # Define features (X) and target (y)
        X = df.drop('faulty', axis=1)
        y = df['faulty']

        # Identify numerical and categorical features
        numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
        categorical_features = ['equipment', 'location']
        
        # Check if features are in the dataset
        if not set(numerical_features).issubset(X.columns) or not set(categorical_features).issubset(X.columns):
            print("Error: Missing required columns in the dataset.")
            return

        # Create preprocessing steps
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessing steps using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Create a pipeline with a preprocessor and a classifier
        # The imblearn Pipeline is used to correctly apply SMOTE
        # on the training data after the split.
        model_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Training model...")
        # Train the model
        model_pipeline.fit(X_train, y_train)
        print("Model training complete.")

        # Save the entire pipeline to a file
        model_filename = 'equipment_anomaly_model.joblib'
        joblib.dump(model_pipeline, model_filename)
        print(f"Model saved to '{model_filename}'")
        
        # You can evaluate the model here to see its performance
        # y_pred = model_pipeline.predict(X_test)
        # from sklearn.metrics import classification_report, confusion_matrix
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))

    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model('equipment_anomaly_data.csv')

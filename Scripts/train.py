import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # SageMaker environment variables for input/output directories
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # File paths for train and validation data
    train_file = os.path.join(input_dir, "train.csv")
    val_file = os.path.join(input_dir, "val.csv")
    
    # Load datasets
    print("Loading training and validation datasets...")
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    
    # Define the pipeline
    print("Defining the pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression())
    ])
    
    # Train the model
    print("Training the model...")
    pipeline.fit(train['text'], train['label'])
    
    # Evaluate the model on validation data
    print("Evaluating the model...")
    val_preds = pipeline.predict(val['text'])
    print(classification_report(val['label'], val_preds))
    
    # Save the trained model to the output directory
    model_path = os.path.join(output_dir, "ml_model.joblib")
    print(f"Saving trained model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    print("Model training and saving complete.")

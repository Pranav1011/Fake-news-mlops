import os
import joblib
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "ml_model.joblib")
    model = joblib.load(model_path)
    if not hasattr(model.named_steps['tfidf'], 'idf_'):
        raise RuntimeError("TfidfVectorizer is not fitted!")
        print("Model verified and loaded.")
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        try:
            print("Raw input received:", request_body)
            return pd.DataFrame({'text': [request_body.strip()]})
        except Exception as e:
            raise ValueError(f"Error parsing input: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    if not isinstance(input_data, pd.DataFrame) or 'text' not in input_data.columns:
        raise ValueError("Input data must be a DataFrame with a 'text' column")
    print("DataFrame input to model:", input_data)
    
    try:
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e
    
    return prediction

def output_fn(prediction, content_type):
    return str(int(prediction[0])) 
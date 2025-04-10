# Fake News Detection MLOps Project on AWS SageMaker

## 🧠 Project Overview
This project implements a real-time fake news detection system using a machine learning pipeline built on AWS. It uses a logistic regression model trained on a dataset of real and fake news headlines. The end-to-end MLOps pipeline was developed entirely using **AWS Free Tier services** to make the solution cost-effective and deployable at scale.

---

## 🔧 Tools & Services Used
- **Amazon SageMaker** – model training, deployment, and endpoint hosting
- **Amazon S3** – data storage for training, validation, and model artifacts
- **scikit-learn 1.2.2** – ML model (TF-IDF + Logistic Regression)
- **Pandas, Joblib** – data handling and model serialization
- **CloudWatch Logs** – monitoring and debugging

---

## ✅ Key Features
- **End-to-End MLOps Pipeline** using AWS SageMaker
- **Real-time inference** with deployed endpoint
- **Free Tier architecture** (no paid AWS services)
- **Custom inference.py** script with robust error handling
- **Confusion matrix and model evaluation** visualized

---

## ⚠️ Errors & Fixes (Technical Debug Summary)

### 1. Model Training Issues
- **Error**: `idf vector is not fitted`
- **Fix**: Ensured `.fit()` was called and verified the trained pipeline locally with `TfidfVectorizer` before uploading to S3.

### 2. Unsupported scikit-learn Version
- **Error**: `ValueError: Unsupported sklearn version`
- **Fix**: Downgraded to `scikit-learn==1.2.2` and used `framework_version='1.2-1'` in SageMaker.

### 3. S3 Path Formatting Error
- **Error**: Double slashes in S3 URI (e.g., `processed//train.csv`)
- **Fix**: Removed trailing slashes in base paths and verified existence using AWS CLI.

### 4. Successful Model Training
- Uploaded `train.csv` and `val.csv` to S3
- Trained directly on SageMaker
- Packaged model as `model.tar.gz` and uploaded to S3

### 5. Model Testing
- **Local Testing**: Extracted `model.tar.gz`, loaded with joblib, ran predictions
- **SageMaker Endpoint**: Successfully deployed and tested using the SageMaker SDK

---

## 🧪 How to Run This Project Locally

### 🔧 Setup
```bash
git clone https://github.com/yourusername/fake-news-mlops.git
cd fake-news-mlops
pip install -r requirements.txt
```

### 🧠 Run Local Model Prediction (Optional)
```bash
python predict_local.py "NASA confirms discovery of water on Mars"
# Output: 1 (Real)
```

---

## 📊 Model Evaluation
- Accuracy: ~95% on test set
- Log Loss: ~0.19

---

## 📤 Future Work
- Integrate with **Streamlit** for a local UI
- Add **API Gateway + Lambda** (free-tier serverless backend)
- Implement **model monitoring** using SageMaker Model Monitor (30 hours free)

---

## 🔗 Connect
Feel free to fork the repo, use the architecture, and ask questions!

---

**Author**: Sai Pranav 


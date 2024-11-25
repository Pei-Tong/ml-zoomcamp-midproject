# ML Zoomcamp Midproject

## Customer Purchase Prediction

### Overview
This project is designed to predict the likelihood of a customer making a purchase based on demographic and behavioral features. It incorporates data preprocessing, exploratory data analysis (EDA), model training, and deployment through a web interface using Streamlit.

The model utilizes a **Random Forest Classifier** to predict purchase probabilities. The project includes local deployment using Docker, with bonus deployment steps for cloud environments.

---

### Features

**Input Features:**
- Age
- Annual Income
- Discounts Availed
- Loyalty Program Status
- Number of Purchases
- Time Spent on Website

**Output:**
- Predicted probability of purchase (a value between 0 and 1).

---

### Project Structure
The project includes the following files:

- **`notebook.py`**:
  - Data preparation and cleaning
  - Exploratory Data Analysis (EDA)
  - Model training and hyperparameter tuning

- **`train.py`**:
  - Training the Random Forest Classifier
  - Saving the trained model using `pickle`

- **`predict.py`**:
  - Loading the model
  - Function to predict customer purchase probabilities

- **`app.py`**:
  - Streamlit-based web application

- **`Dockerfile`**:
  - Instructions for containerizing the application

- **`requirements.txt`**:
  - List of dependencies for the project

- **Dataset**:
  - A CSV file containing the customer dataset (instructions provided to download it if not included).

---

### Prerequisites
- **Python** (version >= 3.8)
- Required Python packages (specified in `requirements.txt`)
- **Docker** (for containerization)
- Dataset available from **[Kaggle: Predict Customer Purchase Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset)**.

---

### Setup and Execution

#### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repository-url/customer-purchase-prediction.git
cd customer-purchase-prediction
```

#### Step 2: Install Dependencies
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate    # For Windows
pip install -r requirements.txt
```

Run the App Locally
```bash
streamlit run app.py
```

## Using Docker
Step 1: Build the Docker Image
```bash
docker build -t customer-purchase-prediction .
Step 2: Run the Docker Container
```bash
docker run -p 8501:8501 customer-purchase-prediction
```

## Project Deliverables
- A Streamlit application for customer purchase prediction.
- Docker container for easy deployment.
- Dataset analysis and preprocessing script in notebook.py.
- Trained model and prediction scripts.

## Web Interface


## Acknowledgements
- Dataset: Kaggle: Predict Customer Purchase Behavior Dataset​.
- Libraries: pandas, numpy, scikit-learn, streamlit, pickle.

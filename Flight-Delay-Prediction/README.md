
# Flight Delay Predictor

A machine learning web application that predicts whether a flight is likely to be delayed based on flight-related information.

The project uses a machine learning model for prediction and provides an interactive interface through Streamlit.

## Features

* Predicts whether a flight will be delayed
* Interactive Streamlit interface
* Data preprocessing and feature handling
* Machine learning-based prediction
* Random Forest-based flight delay prediction
* Simple and user-friendly interface

## Technologies Used

* Python
* Pandas
* Plotly
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

## Machine Learning Models

The project uses the following classification models:

* Random Forest Classifier

The model is trained on flight-related data and evaluated to determine its prediction performance.

## Project Workflow

```text
Flight Dataset
      |
      v
Data Cleaning
      |
      v
Feature Preprocessing
      |
      v
Train/Test Split
      |
      v
Random Forest Training
      |
      v
Model Evaluation
      |
      v
Flight Delay Prediction
      |
      v
Streamlit Dashboard
```

## Project Structure

```text
Flight-Delay-Predictor/
|
├── app.py
├── dataset/
│   └── flight_data.csv
├── ML_Project_Proposal_Airline_Delay (1).docx
├── requirements.txt
└── README.md
```

> The exact file structure may vary depending on the current version of the project.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Flight-Delay-Predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## Objective

The main objective of this project is to use machine learning to predict potential flight delays and demonstrate how a trained classification model can be integrated into an interactive web application.

## Future Improvements

* Improve model accuracy through feature engineering
* Add more flight and weather-related features
* Compare additional machine learning algorithms
* Deploy the application to a cloud platform
* Add real-time flight data integration

## Author

**Maham Jamil**

Data Science Graduate | Python | Machine Learning | Data Engineering

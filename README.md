# Breast Cancer Prediction Using Machine Learning

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

This project implements a machine learning model to predict whether a breast tumor is malignant or benign based on various features extracted from diagnostic images. The model uses the Wisconsin Breast Cancer Dataset and achieves high accuracy in distinguishing between malignant and benign tumors, making it a valuable tool for early cancer detection.

## Features

- Data preprocessing and cleaning
- Extensive Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and selection
- Implementation of multiple machine learning models:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- Model evaluation using various metrics
- Interactive web interface for predictions
- Detailed documentation and analysis notebooks

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aditya3023/breast-cancer-prediction.git
cd breast-cancer-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
breast-cancer-prediction/
│
├── data/                   # Dataset files
│   └── data.csv           # Breast Cancer Wisconsin Dataset
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 1_EDA.ipynb       # Exploratory Data Analysis
│   └── 2_Modeling.ipynb  # Model training and evaluation
│
├── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing functions
│   └── model.py          # Model implementation
│
├── models/               # Saved model files
│   └── model.pkl        # Trained model
│
├── templates/           # Web application templates
│   └── index.html      # Prediction interface
│
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

1. Open and run the Jupyter notebooks in order:
```bash
jupyter notebook notebooks/1_EDA.ipynb
```

2. To make predictions using the trained model:
```python
from src.model import predict_cancer
# Example input features
prediction = predict_cancer(features)
```

3. To run the web interface:
```bash
python src/app.py
```
Then open http://localhost:5000 in your browser.

## Results

The model achieves the following performance metrics:

- Accuracy: 96.5%
- Precision: 95.8%
- Recall: 97.2%
- F1 Score: 96.5%
- ROC-AUC Score: 0.98

Key visualizations and detailed analysis can be found in the notebooks.

## Future Improvements

- Implement additional machine learning algorithms
- Add cross-validation techniques
- Deploy model as a REST API
- Enhance the web interface
- Add more detailed feature importance analysis
- Implement real-time prediction capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Wisconsin Breast Cancer Dataset from UCI Machine Learning Repository
- scikit-learn documentation and community
- All contributors and maintainers

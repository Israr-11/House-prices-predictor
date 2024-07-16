# House Prices Prediction Model

This project contains code for training a machine learning model to predict house prices using the California housing dataset. The model is a Linear Regression model implemented using Scikit-learn.

## Project Structure

- `house_prices_predictor_model.pkl`: The trained machine learning model saved as a pickle file.
- `model_training.ipynb`: Jupyter Notebook containing the code for training the model.
- `.gitignore`: Git ignore file to exclude unnecessary files from the repository.
- `README.md`: This readme file.

## Model Details

The model is a Linear Regression model trained on the California housing dataset. It predicts house prices based on features such as the number of rooms, crime rate, and others.

### Dataset

The dataset used for training is the California housing dataset, which includes the following features:

- `CRIM`: Per capita crime rate by town.
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: Proportion of non-retail business acres per town.
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- `NOX`: Nitric oxides concentration (parts per 10 million).
- `RM`: Average number of rooms per dwelling.
- `AGE`: Proportion of owner-occupied units built prior to 1940.
- `DIS`: Weighted distances to five Boston employment centres.
- `RAD`: Index of accessibility to radial highways.
- `TAX`: Full-value property tax rate per $10,000.
- `PTRATIO`: Pupil-teacher ratio by town.
- `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
- `LSTAT`: Percentage of lower status of the population.

## Model Performance

- **Mean Squared Error (MSE)**: 0.5558915986952444
- **R2 Score**: 0.5757877060324508

## How to Run the Project

### Prerequisites

- Python 3.7+
- Scikit-learn
- Pandas
- NumPy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-prices-prediction-model.git
   cd house-prices-prediction-model


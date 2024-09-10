import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging

# Setup logging
logging.basicConfig(filename='model2_training.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


class DataPreprocessor:
    def __init__(self, url):
        self.url = url
        self.data = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.url)

    def preprocess(self):
        X = self.data.iloc[:, 1:7].values  # Features: X1 to X6
        y = self.data.iloc[:, -1].values  # Target: Y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test


class ModelTrainer:
    def __init__(self, model_class, trials, X_train, y_train):
        self.model_class = model_class
        self.trials = trials
        self.X_train = X_train
        self.y_train = y_train
        self.best_model = None
        self.best_mse = float('inf')
        self.best_errors = []

    def train(self):
        for trial in self.trials:
            alpha = trial["alpha"]
            max_iter = trial["max_iter"]
            tol = trial["tol"]

            # Instantiate the SGDRegressor model with the current trial parameters
            model = self.model_class(alpha=alpha, max_iter=max_iter, tol=tol, random_state=42)

            # Train the model on the training data
            model.fit(self.X_train, self.y_train)

            # Make predictions on the training data
            predictions = model.predict(self.X_train)

            # Calculate performance metrics
            mse = mean_squared_error(self.y_train, predictions)
            r2 = r2_score(self.y_train, predictions)
            explained_variance = explained_variance_score(self.y_train, predictions)
            residuals = self.y_train - predictions
            variance = np.var(residuals)

            # Log trial details
            logging.info(f"Trial with alpha={alpha}, max_iter={max_iter}, tol={tol}")
            logging.info(f"Final theta (parameters): {model.coef_}")
            logging.info(f"Mean Squared Error on training set: {mse}")
            logging.info(f"R2 value on training set: {r2}")
            logging.info(f"Explained Variance on training set: {explained_variance}")
            logging.info(f"Variance of residuals on training set: {variance}")
            logging.info("\n")

            # Print trial details
            print(f"Trial with alpha={alpha}, max_iter={max_iter}, tol={tol}")
            print(f"Final theta (parameters): {model.coef_}")
            print(f"Mean Squared Error on training set: {mse}")
            print(f"R2 value on training set: {r2}")
            print(f"Explained Variance on training set: {explained_variance}")
            print(f"Variance of residuals on training set: {variance}")
            print("")

            # Update the best model if the current model's MSE is lower
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_model = model


class ModelEvaluator:
    def __init__(self, model, X_test, y_test, y_train):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

    def evaluate(self):
        # Make predictions on the test data
        test_predictions = self.model.predict(self.X_test)

        # Calculate performance metrics
        test_mse = mean_squared_error(self.y_test, test_predictions)
        test_r2 = r2_score(self.y_test, test_predictions)
        explained_variance = explained_variance_score(self.y_test, test_predictions)
        test_residuals = self.y_test - test_predictions
        test_variance = np.var(test_residuals)

        # Log test performance metrics
        logging.info("Test performance metrics using the best model parameters from the trials (alpha, max_iter, tol): "
                     f"{self.model.alpha}, {self.model.max_iter}, {self.model.tol}")
        logging.info(f"Mean Squared Error on test set: {test_mse}")
        logging.info(f"R2 value on test set: {test_r2}")
        logging.info(f"Explained Variance on test set: {explained_variance}")
        logging.info(f"Variance of residuals on test set: {test_variance}")
        logging.info("\n")

        # Print the best model parameters and performance metrics
        print("Best model parameters (alpha, max_iter, tol):",
              self.model.alpha, self.model.max_iter, self.model.tol)
        print("Mean Squared Error on test set:", test_mse)
        print("R2 value on test set:", test_r2)
        print("Explained Variance on test set:", explained_variance)
        print("Variance of residuals on test set:", test_variance)

        # Calculate baseline MSE
        baseline_prediction = np.mean(self.y_train)
        baseline_mse = mean_squared_error(self.y_test, np.full_like(self.y_test, baseline_prediction))
        print("Baseline Mean Squared Error:", baseline_mse)

        return test_predictions, test_residuals, baseline_mse


def plot_results(y_test, test_predictions, X_test, important_attribute_index, test_residuals):
    # Plotting the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_predictions, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.show()

    # Plotting the output variable against an important attribute
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, important_attribute_index], y_test, color='blue', label='Actual')
    plt.scatter(X_test[:, important_attribute_index], test_predictions, color='red', label='Predicted')
    plt.xlabel(f'Attribute {important_attribute_index + 1}')
    plt.ylabel('House Prices')
    plt.title(f'House Prices vs Attribute {important_attribute_index + 1}')
    plt.legend()
    plt.show()

    # Plotting the residuals vs fitted values plot
    plt.figure(figsize=(10, 6))
    plt.ylim(-50, 50)
    plt.scatter(test_residuals, test_predictions)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.show()

    # Plotting the histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(test_residuals, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()

    # Normality Q-Q Plot
    plt.figure(figsize=(10, 6))
    stats.probplot(test_residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.show()


if __name__ == "__main__":
    # Data loading and preprocessing
    url = "https://raw.githubusercontent.com/BenTennyson4/real-estate-valuation-dataset/main/Copy%20of%20Real%20estate%20valuation%20data%20set.csv"
    preprocessor = DataPreprocessor(url)
    preprocessor.load_data()
    preprocessor.preprocess()

    # Model training
    trials = [
        #{"alpha": 1e-3, "max_iter": 1000, "tol": 1e-5},
        #{"alpha": 1e-3, "max_iter": 5000, "tol": 1e-5},
        {"alpha": 1e-14, "max_iter": 10000, "tol": 1e-6},
    ]
    trainer = ModelTrainer(SGDRegressor, trials, preprocessor.X_train_scaled, preprocessor.y_train)
    trainer.train()

    # Model evaluation
    evaluator = ModelEvaluator(trainer.best_model, preprocessor.X_test_scaled, preprocessor.y_test,
                               preprocessor.y_train)
    test_predictions, test_residuals, baseline_mse = evaluator.evaluate()

    # Plotting results
    plot_results(preprocessor.y_test, test_predictions, preprocessor.X_test_scaled, important_attribute_index=2,
                 test_residuals=test_residuals)

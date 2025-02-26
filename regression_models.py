import numpy as np
from scipy import stats


class RegressionModel:
    def __init__(self, model_type="constant"):
        self.model_type = model_type
        self.parameters = None
        self.mean = None

    def _convert_to_numeric(self, X):
        """Convert input data to numeric format"""
        if isinstance(X, (list, tuple)):
            # If single predictor value
            if len(X) == 1:
                try:
                    return np.array([[float(x) if isinstance(x, (int, float, str)) else 0 for x in X]])
                except (ValueError, TypeError):
                    return np.array([[0]])
            # If multiple predictor values
            try:
                return np.array([[float(x) if isinstance(x, (int, float, str)) else 0 for x in row] for row in X])
            except (ValueError, TypeError):
                return np.array([[0] * len(X[0])])
        return np.array([[0]])

    def fit(self, X, y):
        """Fit the regression model"""
        try:
            # Convert y to numeric array
            y = np.array([float(val) if isinstance(val, (int, float, str)) else 0 for val in y])

            if self.model_type == "constant":
                self.mean = np.mean(y)
                self.parameters = self.mean

            elif self.model_type == "linear":
                # Convert X to numeric array
                X = self._convert_to_numeric(X)

                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                # Add constant term for intercept
                X = np.column_stack([np.ones(X.shape[0]), X])

                try:
                    # Try normal equation
                    XtX = X.T.dot(X)
                    if np.linalg.det(XtX) != 0:  # Check if matrix is non-singular
                        self.parameters = np.linalg.inv(XtX).dot(X.T).dot(y)
                    else:
                        # Fallback to constant model if matrix is singular
                        self.model_type = "constant"
                        self.mean = np.mean(y)
                        self.parameters = self.mean
                except np.linalg.LinAlgError:
                    # Fallback to constant model if matrix is singular
                    self.model_type = "constant"
                    self.mean = np.mean(y)
                    self.parameters = self.mean

        except Exception as e:
            print(f"Error in model fitting: {str(e)}")
            # Fallback to simple mean
            self.model_type = "constant"
            self.mean = np.mean(y) if len(y) > 0 else 0
            self.parameters = self.mean

    def predict(self, X):
        """Make predictions"""
        try:
            if self.model_type == "constant":
                return np.full(len(X), self.mean)

            elif self.model_type == "linear":
                X = self._convert_to_numeric(X)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X = np.column_stack([np.ones(X.shape[0]), X])
                return X.dot(self.parameters)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return np.full(len(X), self.mean if self.mean is not None else 0)

    def goodness_of_fit(self, y_true, y_pred):
        """
        Calculate goodness of fit using statistical tests as specified in the paper.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            float: p-value for constant model, R² for linear model
        """
        try:
            y_true = np.array([float(val) if isinstance(val, (int, float, str)) else 0 for val in y_true])
            y_pred = np.array([float(val) if isinstance(val, (int, float, str)) else 0 for val in y_pred])

            if len(y_true) < 2:
                return 0.0

            if self.model_type == "constant":
                # Use chi-square test as specified in the paper
                chi_square_stat, p_value = stats.chisquare(y_true, y_pred)
                return p_value  # Higher p-value means better fit

            elif self.model_type == "linear":
                # For linear regression, use both R² and F-test
                n = len(y_true)
                p = len(self.parameters) if hasattr(self, 'parameters') else 1

                # Calculate R-squared
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                ss_res = np.sum((y_true - y_pred) ** 2)

                if ss_tot == 0:
                    return 1.0

                r2 = 1 - (ss_res / ss_tot)

                # Calculate F-statistic
                if n - p > 0:  # Check degrees of freedom
                    f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
                    p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)

                    # Combine R² and p-value
                    # Weight more towards R² but consider p-value
                    combined_score = 0.7 * r2 + 0.3 * (1 - p_value)
                    return max(0.0, combined_score)

                return max(0.0, r2)

        except Exception as e:
            print(f"Error in goodness of fit calculation: {str(e)}")
            return 0.0

    def predict_func(self, x):
        if isinstance(x, (list, tuple)):
            return self.predict([x])[0]
        return self.predict([[x]])[0]

    @staticmethod
    def create_regression_function(model_type, X, y):
        """Create a regression function"""
        model = RegressionModel(model_type)
        model.fit(X, y)

        def regression_func(x):
            return model.predict([x])[0]

        return regression_func
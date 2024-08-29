from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

class RegressionModels:
    def __init__(self, n_estimators=100, random_state=42, test_size=0.2):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.models = {}

    def train_models(self, X, y):
        """Train Random Forest models for all target variables."""
        print("Training Random Forest models...")
        results = {}

        for col in y.columns:
            print(f"Processing activity: {col}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[col], test_size=self.test_size, random_state=self.random_state
            )

            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y[col], cv=5, scoring='neg_mean_squared_error')
            results[col] = {
                'MSE': mse,
                'R2': r2,
                'CV_MSE': -cv_scores.mean(),
                'CV_MSE_std': cv_scores.std()
            }
            self.models[col] = model
        return results

    def save_models(self, output_dir):
        """Save trained models to disk."""
        for col, model in self.models.items():
            model_filename = f"{output_dir}/rf_model_{col}.joblib"
            dump(model, model_filename)
            print(f"Model for {col} saved to {model_filename}")

    def get_feature_importances(self):
        """Get feature importances for all models."""
        importances = {}
        for col, model in self.models.items():
            importances[col] = model.feature_importances_
        return importances
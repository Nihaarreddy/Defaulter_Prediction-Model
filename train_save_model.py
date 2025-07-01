
 
import joblib
 
def train_and_save_final_model(X, y, model, model_name="final_model.pkl"):

    """

    Train the best model on full data and save it to a file.

    Parameters:

    - X: Feature DataFrame

    - y: Target Series

    - model: Best estimator returned from grid search

    - model_name: File path to save the model

    """

    print(" Training the best model on full dataset...")

    model.fit(X, y)

    joblib.dump(model, model_name)

    print(f" Model saved as: {model_name}")
 
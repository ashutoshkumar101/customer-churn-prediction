import json
import joblib
from pathlib import Path

def save_artifacts(model, preprocessor, threshold, save_dir="models"):
    """
    Save the trained model, preprocessing pipeline, and deployment configuration
    """

    save_path= Path(save_dir)
    save_path.mkdir(parents= True, exist_ok= True)

    # Save model
    joblib.dump(model, save_path / "logistic_regression.pkl")

    # Save preprocessing pipeline
    joblib.dump(preprocessor, save_path / "preprocessing_pipeline.pkl")

    # Save configuration
    config= {
        "threshold": threshold
    }

    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Artifacts saved successfully in '{save_path}'.")
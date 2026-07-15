from src.evaluation import evaluate_model

def run_experiment(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name
):
    model.fit(X_train, y_train)

    results=  evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name
    )

    return results
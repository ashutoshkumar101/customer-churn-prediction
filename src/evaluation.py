from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name
):
    positive_label= 1 if 1 in y_test.unique() else "Yes"

    y_train_pred= model.predict(X_train)
    y_test_pred= model.predict(X_test)
    y_test_pred_proba= model.predict_proba(X_test)[:,1]

    train_acc= accuracy_score(y_train, y_train_pred)
    test_acc= accuracy_score(y_test, y_test_pred)

    precision= precision_score(y_test, y_test_pred, pos_label=positive_label)
    recall= recall_score(y_test, y_test_pred, pos_label=positive_label)
    f1= f1_score(y_test, y_test_pred, pos_label= positive_label)
    roc_auc= roc_auc_score(y_test, y_test_pred_proba)

    print(f"===========================\n{model_name}\n===========================\n")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification Report: \n",classification_report(y_test, y_test_pred))
    print("\nConfusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
    print(f"\nROC-AUC: {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Train accuracy": train_acc,
        "Test accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

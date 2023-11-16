from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    balanced_accuracy_score,
    average_precision_score,
    PrecisionRecallDisplay,
    confusion_matrix
)
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from data_preprocessing import prep_data

# fit classifier with CV best params
clf = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=100,
    max_depth=15,
    l2_regularization=0,
    learning_rate=1
)

metric = 'balanced_accuracy'

if __name__ == "__main__":

    train, _ = prep_data()
    train_size = int(train.shape[0]*0.8)

    X_train = train.iloc[:train_size].drop(columns='accept')
    y_train = train['accept'].iloc[:train_size]
    X_test = train.iloc[train_size:].drop(columns='accept')
    y_test = train['accept'].iloc[train_size:]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # write metrics to file
    with open('tables_charts/model_metrics.txt', 'w') as f:
        print(classification_report(y_test, y_pred), '\n', file=f)
        print(f'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}', '\n', file=f)
        print(f'Average Precision: {average_precision_score(y_test, y_proba)}', '\n', file=f)
        print(f'AUC: {roc_auc_score(y_test, y_proba)}', '\n', file=f)

    # plot precision-recall curve
    display = PrecisionRecallDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        plot_chance_level=True
    )
    _ = display.ax_.set_title("Precision-Recall Curve")
    plt.savefig('tables_charts/precision_recall_curve.png')

    # plot ROC curve
    display = RocCurveDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        plot_chance_level=True
    )
    _ = display.ax_.set_title("ROC Curve")
    plt.savefig('tables_charts/roc_curve.png')

    # determine permutation feature importance
    result = permutation_importance(
        clf,
        X_train,
        y_train,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring=metric
    ) 

    # plot mean decrease in score for each permuted feature in HGB classifier
    threshold = 0.005
    clf_importances = pd.Series(result.importances_mean, index=clf.feature_names_in_)
    fig, ax = plt.subplots()
    clf_importances[clf_importances > threshold].plot.bar(yerr=result.importances_std[clf_importances > threshold], ax=ax)
    ax.set_title("Feature Importances using Permutation")
    ax.set_ylabel(f"Mean {metric} decrease")
    fig.tight_layout()
    plt.savefig('tables_charts/feature_importances.png')

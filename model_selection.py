from data_preprocessing import prep_data, create_imputer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

input, _ = prep_data()
train_size = int(input.shape[0]*0.8)

# input dataset train-test split based on order_day
X_train = input.iloc[:train_size].drop(columns='accept')
y_train = input['accept'].iloc[:train_size]
X_test = input.iloc[train_size:].drop(columns='accept')
y_test = input['accept'].iloc[train_size:]

# impute based on training set
imputer = create_imputer(X_train=X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# time-based cross validation splits
tscv = TimeSeriesSplit(n_splits=5)

# enumerate models and parameters to explore and evaluate
rfc = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rfc_param_grid = {
    'ccp_alpha': np.linspace(0, 1, 9),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': np.arange(2, 50, 2)
}

hgb = HistGradientBoostingClassifier(
    class_weight='balanced',
    scoring='f1',
    random_state=42
)
hgb_param_grid = {
    'max_depth': np.arange(2, 50, 2),
    'max_iter': [100, 1000],
    'l2_regularization': np.linspace(0, 100, 11),
    'learning_rate': np.linspace(0.1, 1, 10),
}

knn = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': np.arange(3, 6),
    'weights': ['uniform', 'distance'],
    'p': np.arange(1, 3)
}

log = LogisticRegression(
    class_weight='balanced',
    solver='saga',
    random_state=42
)
log_param_grid = {
     'C': np.linspace(0.0001, 1, 1000),
     'penalty': ['l1', 'l2', 'elasticnet'],
     'l1_ratio': np.linspace(0, 1, 10)
}

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42
)
xgb_param_grid = {
    'max_depth': np.arange(2, 50, 2),
    'grow_policy': ['depthwise', 'lossguide'],
    'learning_rate': np.linspace(0.1, 1, 10),
    'tree_method': ['hist', 'exact', 'approx']
}

# combine models, params and metrics into list for zip and iteration
model_names = [
    'rfc',
    'hgb',
    'knn',
    'log',
    'xgb'
]

models = [
    rfc,
    hgb,
    knn,
    log,
    xgb
]

params = [
    rfc_param_grid,
    hgb_param_grid,
    knn_param_grid,
    log_param_grid,
    xgb_param_grid
]

metrics = ['balanced_accuracy', 'f1', 'average_precision']

# placeholder DataFrame to populate CV final results
df_cv_best_results = pd.DataFrame()
cv_cols = [
    'model_name',
    'params',
    'mean_test_balanced_accuracy',
    'std_test_balanced_accuracy',
    'mean_test_f1',
    'std_test_f1',
    'mean_test_average_precision',
    'std_test_average_precision'
]

if __name__ == "__main__":

    # CV search across all models using randomly selected params
    for name, clf, param in zip(model_names, models, params):
        cross_val = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param,
            scoring=metrics,
            cv=tscv.split(X_train),
            n_jobs=-1,
            n_iter=25,
            random_state=42,
            refit=False
        )
        cross_val.fit(X_train, y_train)

        # track results for each model
        result = pd.DataFrame(cross_val.cv_results_)
        result['model_name'] = name
        df_cv_best_results = pd.concat([df_cv_best_results, result[cv_cols]])

    cv_summary = pd.DataFrame(
        index=model_names,
        columns=metrics
    )
    cv_summary.index.name = 'model'

    # record the best performing model/params across all metrics
    for name in model_names:
        for m in metrics:
            top_score = df_cv_best_results[df_cv_best_results['model_name'] == name][f'mean_test_{m}'].max()
            cv_summary.loc[name][m] = np.round(top_score, 3)

    cv_summary.to_csv('tables_charts/model_selection_cv_results.csv')

from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
from data_preprocessing import prep_data
import pickle

# fit classifier with CV best params
clf = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=100,
    max_depth=15,
    l2_regularization=0,
    learning_rate=1
)

if __name__ == "__main__":
    
    train, X_test = prep_data()
    X_train = train.drop(columns='accept')
    y_train = train['accept']

    clf.fit(X_train, y_train)
    output = clf.predict_proba(X_test)[:, 1]

    with open('deliverables/fayaaz_khatri_predictions.pkl', 'wb') as p:
        pickle.dump(output, p)

# Environment Setup

This project uses Python 3.9.12. The following commands will create a virtual environment in the top-level directory of the project called `venv`. The virtual environment will be activated and required dependencies will be installed.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# File Structure

```
.
├── README.md
├── data
│   ├── ecoshare_sales_test.csv
│   └── ecoshare_sales_v3.xlsx
├── data_preprocessing.py
├── deliverables # *** add final paper filename here ***
│   ├── Fayaaz_Khatri_Midterm_Progress.pdf
│   └── test_set_predicted_proba.pickle
├── model_evaluation.py
├── model_output.py
├── model_selection.py
├── requirements.txt
└── tables_charts
    ├── feature_importances.png
    ├── model_metrics.txt
    ├── model_selection_cv_results.csv
    ├── precision_recall_curve.png
    └── roc_curve.png
```

# Deliverables

Predicted probabilities for the test set are provided in a pickle file. Use the below snippet to load the pickle file in a script.

```python
import pickle

with open('deliverables/fayaaz_khatri_predictions.pkl', 'rb') as p:
    output = pickle.load(p)
```
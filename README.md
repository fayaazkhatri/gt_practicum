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
│   ├── ecoshare_sales_test.csv
│   └── ecoshare_sales_v3.xlsx
├── data_exploration.py
├── data_preprocessing.py
├── deliverables
│   ├── Fayaaz_Khatri_Final_Report.pdf
│   ├── Fayaaz_Khatri_Midterm_Progress.pdf
│   └── fayaaz_khatri_predictions.pkl
├── model_evaluation.py
├── model_output.py
├── model_selection.py
├── requirements.txt
└── tables_charts
    ├── confusion_matrix.png
    ├── feature_importances.png
    ├── model_metrics.txt
    ├── model_selection_cv_results.csv
    ├── precision_recall_curve.png
    ├── quarterly_call_volume.png
    ├── quarterly_conversions.png
    └── roc_curve.png
```

# Deliverables

Predicted probabilities for the test set are provided in a pickle file. Use the below snippet to load the pickle file in a script.

```python
import pickle

with open('deliverables/fayaaz_khatri_predictions.pkl', 'rb') as p:
    output = pickle.load(p)
```

The final report is stored at `deliverables/Fayaaz_Khatri_Final_Report.pdf`.

# Scripts

To train the model and predict against the test set, run the following command. A `pickle` file will be written to the `deliverables` subdirectory.

```bash
python model_output.py
```

The followings scripts correspond to sections of the final paper. Feel free to run them to recreate items in the `tables_charts` subdirectory.

```bash
python data_exploration.py
python data_preprocessing.py
python model_selection.py
python model_evaluation.py
```
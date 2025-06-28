import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('./data/enriched/HB.csv', sep=',')
tpot_data = tpot_data.rename(columns={'occurrence': 'target'})
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.95673027737156
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=3, min_child_weight=8, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)),
    BernoulliNB(alpha=10.0, fit_prior=False)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Fit and predict
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Evaluate
print("TPOT Pipeline Test Accuracy:", accuracy_score(testing_target, results))
print(classification_report(testing_target, results))
print(confusion_matrix(testing_target, results))
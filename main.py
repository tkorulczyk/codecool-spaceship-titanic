import numpy as np
import optuna
import pandas as pd

# nn
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam

# Gradient Boosting
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Local imports
from preprocessor.preprocess import preprocess
from utils.utils import print_metrics, print_crossvalidated_metrics
from tunner.params_tunner import objective

# Set warnings and logging
import subprocess
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Flag for tuning
show_confusion_matrices = False
submit_to_kaggle = False
running_mode = 'standard' # Running modes: ['tunner', 'nn', 'standard']


# 1. Data import
df_train = pd.read_csv('train.csv', index_col=False)
df_test = pd.read_csv('test.csv', index_col=False)

df_train['SetId'] = 'Train'
df_test['SetId'] = 'Test'
target_column = 'Transported'
df = pd.concat([df_train, df_test]).copy()
original_columns = df.columns
catb_best_params = None

# 2. Preprocess and tune train data
X, y, X_train, X_val, y_train, y_val, X_test, passenger_ids = preprocess(df, target_column)


def study_and_print(model_type):
    study.optimize(lambda trial: objective(trial, X_train, y_train, model_type), n_trials=50)
    # Display best parameters and accuracy
    print(f"Best parameters for {model_type}: {study.best_params}")
    print(f"Best accuracy value for {model_type}: {study.best_value}")


# 3. Hyperparameter tuning and model fitting
if running_mode == 'tunner':
    model_type = ''
    # Initialize Optuna study object
    study = optuna.create_study(direction='maximize', study_name=f"{model_type} Hyperparameter Optimization")

    study_and_print('catboost')
    study_and_print('lightgbm')
    study_and_print('xgboost')

elif running_mode == 'nn':
    # input for numerical characteristics
    numeric_input = Input(shape=(6,))
    dense_layer = Dense(10, activation='relu')(numeric_input)

    # input for categorical characteristics
    categorical_inputs = []
    num_categories = [len(np.unique(X_train[:, i])) for i, col in enumerate(original_columns) if
                      df[col].dtype == 'object']
    numeric_data = X_train[:, [i for i, col in enumerate(original_columns) if df[col].dtype != 'object']]
    categorical_data = [X_train[:, i] for i, col in enumerate(original_columns) if df[col].dtype == 'object']

    for i in range(13):
        categorical_input = Input(shape=(1,))
        embedding = Embedding(input_dim=num_categories[i], output_dim=10, input_length=1)(categorical_input)
        categorical_inputs.append(categorical_input)
        dense_layer = Concatenate()([dense_layer, Flatten()(embedding)])

    # połączenie warstw
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model([numeric_input] + categorical_inputs, output_layer)

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit([numeric_data] + categorical_data, y_train, epochs=10, batch_size=32)
    catb = CatBoostClassifier(verbose=0)

    ŷ_train = catb.predict(X_train)
    ŷ_val = catb.predict(X_val)

    result = print_metrics('CatBoost', Train=(y_train, ŷ_train), Val=(y_val, ŷ_val))

    results = []
    results.append(result)

    results_df = pd.DataFrame(results)

    print("Results for Train set:")
    print(results_df.filter(like="Train"))

    print("\nResults for Val set:")
    print(results_df.filter(like="Valid"))

elif running_mode == 'standard':
    # CatBoost params
    catb_parameters = {
        'bootstrap_type': 'Bayesian',
        'iterations': 1885,
        'depth': 14,
        'min_data_in_leaf': 162,
        'l2_leaf_reg': 4,
        'rsm': 0.8784357141833647,
        'early_stopping_rounds': 11,
        'od_type': 'IncToDec',
        'learning_rate': 0.02888812191896493,
        'objective': 'Logloss',
        'eval_metric': 'Accuracy',
        'grow_policy': 'SymmetricTree',
        'leaf_estimation_iterations': 1,
        'boosting_type': 'Ordered',
        'leaf_estimation_method': 'Newton',
        'auto_class_weights': 'Balanced',
        'bagging_temperature': 0.03986831910447205
    }

    catb = CatBoostClassifier(**catb_best_params) if catb_best_params else CatBoostClassifier(**catb_parameters)

    models = [
        ('XGBoost', XGBClassifier()),
        ('LightGBM', LGBMClassifier()),
        ('CatBoost', catb)
    ]

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'jaccard', 'roc_auc', 'matthews_corrcoef']

    print_crossvalidated_metrics(models, scoring, X_train, y_train, X_val, y_val, X, y,
                                 show_confusion_matrices=show_confusion_matrices)

    catb.fit(X_train, y_train, verbose=False)

# 5. Predicting and creating submission
y_test_pred = catb.predict(X_test)
y_test_pred = np.array(list(map(lambda x: True if x == 1 else False, y_test_pred)))

if submit_to_kaggle:
    # Using previously saved indices as PassengerId
    result = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': y_test_pred})
    file_name = 'submission.csv'
    result.to_csv(file_name, index=False)

    # 6. Submitting to Kaggle
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", "spaceship-titanic", "-f", file_name, "-m", "Another trial"])

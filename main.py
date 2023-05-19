import subprocess
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

# Local imports
from preprocessing import preprocess

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Filter warnings
warnings.filterwarnings('ignore')

# Set optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Flag for tuning
tuned = True
submit_to_kaggle = True

# 1. Data import
df_train = pd.read_csv('train.csv', index_col=False)
df_test = pd.read_csv('test.csv', index_col=False)

df_train['SetId'] = 'Train'
df_test['SetId'] = 'Test'
target_column = 'Transported'
df = pd.concat([df_train, df_test]).copy()

# 2. Preprocess and tune train data
# X_train, X_val, y_train, y_val = preprocess(df)
X_train, X_val, y_train, y_val, X_test, passenger_ids = preprocess(df, target_column)

# 4. Hyperparameter tuning and model fitting
if tuned:
    def objective(trial):
        """
        This function defines the hyperparameter space to be searched and optimizes them using Optuna.
        """
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'depth': trial.suggest_int('depth', 1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 200),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5),
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 0.2),
            'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
            'rsm': trial.suggest_uniform('rsm', 0.8, 1.0),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 5, 30),
            'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'objective': trial.suggest_categorical("objective", ['Logloss']),
            'eval_metric': trial.suggest_categorical('eval_metric', ['Accuracy']),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced']),

            'task_type': 'GPU',
            'devices': 'GPU:1',
            'verbose': False
        }

        # Odfiltrowuj niedozwolone kombinacje parametrów
        if params['boosting_type'] == 'Ordered' and params['grow_policy'] in ['Depthwise', 'Lossguide']:
            raise optuna.TrialPruned("Ordered boosting is not supported for nonsymmetric trees")

        # if params['boosting_type'] == 'Ordered' and params['grow_policy'] in ['Depthwise', 'Lossguide']:
        #     raise optuna.TrialPruned("Ordered boosting is not supported for nonsymmetric trees")

        model = CatBoostClassifier(**params, random_seed=42)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        accuracy = score.mean()

        return accuracy


    # Initialize study object and conduct optimization
    study = optuna.create_study(direction='maximize', study_name="CatBoost Hyperparameter Optimization")
    study.optimize(objective, n_trials=100)

    # Display best parameters and accuracy
    print("Best parameters:", study.best_params)
    print("Best accuracy value:", study.best_value)

    # Training the model with the best parameters
    catb = CatBoostClassifier(**study.best_params, verbose=0)
    catb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
else:
    # Training the model without tuning
    catb = CatBoostClassifier(verbose=0)
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

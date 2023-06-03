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
tuned = False
nn = False
submit_to_kaggle = True

# 1. Data import
df_train = pd.read_csv('train.csv', index_col=False)
df_test = pd.read_csv('test.csv', index_col=False)

df_train['SetId'] = 'Train'
df_test['SetId'] = 'Test'
target_column = 'Transported'
df = pd.concat([df_train, df_test]).copy()
original_columns = df.columns
# original_columns.drop(columns=['SetId', 'Name', 'DeckNum'], axis=1, inplace=True)

# 2. Preprocess and tune train data
# X_train, X_val, y_train, y_val = preprocess(df)
X_train, X_val, y_train, y_val, X_test, passenger_ids = preprocess(df, target_column)

# 4. Hyperparameter tuning and model fitting
if tuned:
    def objective(trial):
        """
        This function defines the hyperparameter space to be searched and optimizes them using Optuna.
        """
        bootstrap_type_value = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'Bayesian', 'No'])
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'depth': trial.suggest_int('depth', 1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 200),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5),
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
            'bootstrap_type': bootstrap_type_value,
            'verbose': False,

        }

        if bootstrap_type_value == 'Bernoulli':
            params['subsample'] = trial.suggest_uniform('subsample', 0.8, 1.0)
        elif bootstrap_type_value == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 0.2)
        elif bootstrap_type_value == 'No':
            if 'bagging_temperature' in params:
                del params['bagging_temperature']
            if 'subsample' in params:
                del params['subsample']

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

elif nn:
    from keras.models import Model
    from keras.layers import Input, Dense, Embedding, Concatenate, Flatten
    from keras.optimizers import Adam

    # inputy dla cech numerycznych
    numeric_input = Input(shape=(6,))
    dense_layer = Dense(10, activation='relu')(numeric_input)

    # inputy dla cech kategorycznych
    categorical_inputs = []
    num_categories = [len(np.unique(X_train[:, i])) for i, col in enumerate(original_columns) if
                      df[col].dtype == 'object']
    numeric_data = X_train[:, [i for i, col in enumerate(original_columns) if df[col].dtype != 'object']]
    categorical_data = [X_train[:, i] for i, col in enumerate(original_columns) if df[col].dtype == 'object']

    for i in range(13):  # liczba kolumn kategorycznych
        categorical_input = Input(shape=(1,))
        embedding = Embedding(input_dim=num_categories[i], output_dim=10, input_length=1)(categorical_input)
        categorical_inputs.append(categorical_input)
        dense_layer = Concatenate()([dense_layer, Flatten()(embedding)])

    # połączenie warstw
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model([numeric_input] + categorical_inputs, output_layer)

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # teraz możemy dopasować model
    model.fit([numeric_data] + categorical_data, y_train, epochs=10, batch_size=32)

else:
    # Training the model without tuning
    # catb = CatBoostClassifier(verbose=0)
    # catb.fit(X_train, y_train, verbose=False)

    catb = CatBoostClassifier(bootstrap_type='Bayesian',
                              iterations=1885, depth=14,
                              min_data_in_leaf= 162,
                              l2_leaf_reg= 4,
                              rsm= 0.8784357141833647,
                              early_stopping_rounds= 11,
                              od_type= 'IncToDec',
                              learning_rate= 0.02888812191896493,
                              objective= 'Logloss',
                              eval_metric= 'Accuracy',
                              grow_policy= 'SymmetricTree',
                              leaf_estimation_iterations= 1,
                              boosting_type= 'Ordered',  # Fix typo here
                              leaf_estimation_method= 'Newton',
                              auto_class_weights= 'Balanced',
                              bagging_temperature= 0.03986831910447205)
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

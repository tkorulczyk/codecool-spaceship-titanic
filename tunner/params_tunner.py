from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold



def train_and_evaluate_model(best_params, X_train, y_train, X_val, y_val):
    catb = CatBoostClassifier(**best_params, verbose=False)

    # Setting up early stopping
    params = {
        'eval_set': (X_val, y_val),
        'early_stopping_rounds': 30,
        'verbose': False
    }

    catb.fit(X_train, y_train, **params)
    y_pred_train = catb.predict(X_train)
    y_pred_val = catb.predict(X_val)

    return y_pred_train, y_pred_val


def objective(trial, X_train, y_train, model_type):
    def get_model_params(trial, model_type):
        if model_type == 'catboost':
            bootstrap_type_value = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'Bayesian', 'No'])
            params = {
                # Hypertuned params
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'depth': trial.suggest_int('depth', 1, 15),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),

                # Loss function management
                'objective': trial.suggest_categorical("objective", ['Logloss']),
                'eval_metric': trial.suggest_categorical('eval_metric', ['Accuracy']),

                # Overfitting management
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 5, 30),
                'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced']),

                # Tree structure and growth management
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 200),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),

                # Regularization and Sampling
                'rsm': trial.suggest_uniform('rsm', 0.8, 1.0),
                'bootstrap_type': bootstrap_type_value,
                'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 0.2),
                'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),

                # Boosting scheme management
                'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),

                # Miscellaneous parameters
                'task_type': 'CPU',
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

            model = CatBoostClassifier(**params, random_seed=42)

        elif model_type == 'xgboost':
            params = {
                # Number of boosting iterations
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),

                # Learning rate, controls the contribution of each tree in the ensemble
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),

                # Tree complexity and growth management
                'max_depth': trial.suggest_int('max_depth', 1, 15),
                # maximum depth of the tree, used to control over-fitting

                'gamma': trial.suggest_uniform('gamma', 0, 1),
                # minimum loss reduction required to make a further partition on a leaf node of the tree, acts as regularization

                # Data sampling
                'subsample': trial.suggest_uniform('subsample', 0.5, 1),
                # fraction of the training instances to be randomly samples for each boosting round, used to prevent overfitting
            }

            model = XGBClassifier(**params, random_seed=42, use_label_encoder=False, eval_metric='mlogloss')

        elif model_type == 'lightgbm':
            params = {
                # Number of boosting iterations
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),

                # Learning rate, controls the contribution of each tree in the ensemble
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),

                # Tree complexity and growth management
                'max_depth': trial.suggest_int('max_depth', 1, 15),
                # maximum depth of the tree, used to control over-fitting

                'num_leaves': trial.suggest_int('num_leaves', 2, 256),  # maximum number of leaves in one tree
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),

            }
            model = LGBMClassifier(**params, random_seed=42)

        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        return model

    model = get_model_params(trial, model_type)

    skf = StratifiedKFold(n_splits=10)
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    accuracy = score.mean()

    return accuracy

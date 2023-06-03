# Standard library imports
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Scikit-learn & third-party classifier imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer

# Scikit-learn metrics imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, \
    matthews_corrcoef

# Scikit-learn preprocessing imports & model selection imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabulate import tabulate
from xgboost import XGBClassifier


def get_family_id(df):
    df['FamilyId'] = None
    group_family_counter = 1
    previous_group_id = -1
    group_counts = df['GroupId'].value_counts()
    for index, row in df.iterrows():
        same_group = df[df['GroupId'] == row['GroupId']]
        same_last_name_in_group = same_group[same_group['LastName'] == row['LastName']]

        if row['GroupId'] != previous_group_id:
            group_family_counter = 1
            previous_group_id = row['GroupId']

        if group_counts[row['GroupId']] > 1 and len(same_last_name_in_group) > 1:
            if pd.isnull(df.at[index, 'FamilyId']):
                family_members = same_last_name_in_group.index
                for member_index in family_members:
                    df.at[member_index, 'FamilyId'] = group_family_counter
                group_family_counter += 1
        elif pd.isnull(df.at[index, 'FamilyId']):
            df.at[index, 'FamilyId'] = 0
    # df.drop(['LastName', 'FirstName'], axis=1, inplace=True)

    return df


def one_hot_encoding(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='.')
    return df_encoded


def num_inputation(df, n_iterations=10):
    missing_vals = df.isna()
    # Get all numerical columns from df
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Iterate over each numerical column
    for numerical_column in numerical_columns:
        # Przechowywanie wyników imputacji dla każdej iteracji
        imputed_data_results = []
        for _ in range(n_iterations):
            imputer = IterativeImputer(n_nearest_features=10, random_state=42)
            imputed_data = imputer.fit_transform(df[numerical_columns])
            imputed_data_results.append(imputed_data)

        # Convert the result list into a 3D numpy array
        imputed_data_results = np.array(imputed_data_results)

        # Get the column index of the current numerical column
        column_index = numerical_columns.index(numerical_column)

        # Obliczanie średniej ze wszystkich impotacji dla danej kolumny
        imputed_data_mean = np.mean(imputed_data_results[:, :, column_index], axis=0)

        # Zastępowanie brakujących wartości średnią wartością dla każdego brakującego elementu
        missing_values_in_column = missing_vals[numerical_column]
        df.loc[missing_values_in_column, numerical_column] = imputed_data_mean[missing_values_in_column]

    return df


def split_column(df, column_name, separator, new_column_names):
    """
    This function splits a column into multiple columns based on a separator.

    Parameters:
    df (pd.DataFrame): The DataFrame to manipulate.
    column_name (str): The name of the column to split.
    separator (str): The separator to use for the split.
    new_column_names (list): The names of the new columns.

    Returns:
    df (pd.DataFrame): The DataFrame with the new columns.
    """

    df[new_column_names] = df[column_name].str.split(separator, expand=True)
    df.drop(columns=column_name, inplace=True)

    return df


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate and return various evaluation metrics for given ground truth labels and predicted labels.

    Parameters:
    y_true (array-like): Ground truth labels.
    y_pred (array-like): Predicted labels.
    average (str, optional): The averaging method to be used for calculating precision, recall, and F1-score.
                              Defaults to 'weighted'.

    Returns:
    tuple: A tuple containing the following metrics (in order): accuracy, precision, recall, F1-score,
           Jaccard score, ROC AUC score, and Matthews correlation coefficient.
    """
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average=average), 3)
    recall = round(recall_score(y_true, y_pred, average=average), 3)
    f1 = round(f1_score(y_true, y_pred, average=average), 3)
    jaccard = round(jaccard_score(y_true, y_pred, average=average), 3)
    roc_auc = round(roc_auc_score(y_true, y_pred), 3)
    matthews_corr = round(matthews_corrcoef(y_true, y_pred), 3)
    return accuracy, precision, recall, f1, jaccard, roc_auc, matthews_corr


def print_metrics(model_name, y_train_true, y_train_pred, y_val_true, y_val_pred):
    metrics_train = compute_metrics(y_train_true, y_train_pred)
    metrics_val = compute_metrics(y_val_true, y_val_pred)
    mislabeled = (y_val_true != y_val_pred).sum()
    result = {
        'Model': model_name,
        'Mislabeled': mislabeled,
        'Accuracy Train': metrics_train[0],
        'Accuracy Val': metrics_val[0],
        'Precision Train': metrics_train[1],
        'Precision Val': metrics_val[1],
        'Recall Train': metrics_train[2],
        'Recall Val': metrics_val[2],
        'F1 Train': metrics_train[3],
        'F1 Val': metrics_val[3],
        'Jaccard Train': metrics_train[4],
        'Jaccard Val': metrics_val[4],
        'ROC AUC Train': metrics_train[5],
        'ROC AUC Val': metrics_val[5],
        'Matthews Corr Train': metrics_train[6],
        'Matthews Corr Val': metrics_val[6],
    }
    return result


def fit_and_predict(model, X_train, y_train, X_val):
    """
    Fit the given model with the training data and make predictions for both training and validation sets.

    Parameters:
    model: A classifier model instance.
    X_train (array-like): The training input samples.
    y_train (array-like): The target values for the training input samples.
    X_val (array-like): The validation input samples.

    Returns:
    tuple: A tuple containing the predicted labels for the training set and the validation set.
    """
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    return y_train_pred, y_val_pred


def analyze_data(df):
    categorical_var_names, numerical_var_names = get_column_types(df)

    # Prepare empty DataFrames for storing summary statistics
    categorical_summary = pd.DataFrame()
    numerical_summary = pd.DataFrame()
    missing_data = df.isnull().sum().to_frame().T
    missing_data.index = ['MISSING DATA']

    for col in df.columns:
        if col in categorical_var_names and col != 'Name' and col != 'PassengerId' and col != 'Cabin':
            counts = df[col].value_counts().to_frame()
            counts.columns = [f'{col} [freq]']
            # Calculate percentage and format it as an integer with a % sign
            percentages = (df[col].value_counts(normalize=True) * 100).apply(lambda x: f'{int(x)}%').to_frame()
            percentages.columns = [f'{col} [%]']
            # Concatenate count and percentage dataframes horizontally
            counts_and_percentages = pd.concat([counts, percentages], axis=1)
            categorical_summary = pd.concat([categorical_summary, counts_and_percentages], axis=1)
        elif col in numerical_var_names:
            descriptives = df[col].describe().to_frame().T
            numerical_summary = pd.concat([numerical_summary, descriptives])

    # Display the categorical and numerical summaries
    if not categorical_summary.empty:
        categorical_summary = pd.concat([missing_data, categorical_summary])
        print("---- CATEGORICAL SUMMARY ----")
        print(tabulate(categorical_summary, headers='keys', tablefmt='psql', showindex=True))
    else:
        print("No categorical data to display.")

    if not numerical_summary.empty:
        # numerical_summary = pd.concat([missing_data, numerical_summary])
        print("---- NUMERICAL SUMMARY ----")
        print(tabulate(numerical_summary, headers='keys', tablefmt='psql', showindex=True))
    else:
        print("No numerical data to display.")


def get_column_types(df_train, acceptable_duplication_threshold=0.05):
    numerical_var_names = df_train._get_numeric_data().columns
    categorical_var_names = list(df_train.select_dtypes(include=['object']).columns)

    for col in categorical_var_names[:]:
        unique_values_percent = df_train[col].nunique() / len(df_train)
        if unique_values_percent > acceptable_duplication_threshold:
            categorical_var_names.remove(col)

    return categorical_var_names, numerical_var_names


def fill_missing_vip(df):
    """
    Fill missing VIP values based on age and HomePlanet conditions.
    """
    df['VIP'] = np.where(df['VIP'].isnull() & (df.Age < 25), False, df.VIP)
    df['VIP'] = np.where(df['VIP'].isnull() & df.HomePlanet.str.contains('Earth'), False, df['VIP'])
    df['VIP'] = np.where(df['VIP'].isnull() & df.HomePlanet.str.contains('Europa'), True, df['VIP'])

    return df


def fill_missing_homeplanet(df):
    """
    Fill missing HomePlanet values based on DeckNo and VIP conditions.
    """
    conditions = [
        (df.HomePlanet.isnull() & df["DeckNo"].str.contains('G')),
        (df.HomePlanet.isnull() & df["DeckNo"].str.contains('B')),
        (df.HomePlanet.isnull() & df["DeckNo"].str.contains('A')),
        (df.HomePlanet.isnull() & df["DeckNo"].str.contains('C')),
        (df.HomePlanet.isnull() & df['VIP'] & df["DeckNo"].str.contains('F')),
        (df.HomePlanet.isnull() & (df['VIP'] == False)),
        (df.HomePlanet.isnull() & df.CryoSleep & df.VIP)
    ]
    values = ['Earth', 'Europa', 'Europa', 'Europa', 'Mars', 'Earth', 'Europa']

    df['HomePlanet'] = np.select(conditions, values, default=df.HomePlanet)
    return df


def fill_missing_cryosleep(df, expenses):
    """
    Fill missing CryoSleep values based on expenses conditions.
    """
    df['CryoSleep'] = np.where(df.CryoSleep.isnull() & (expenses == 0), True, df.CryoSleep)
    df['CryoSleep'] = np.where(df.CryoSleep.isnull() & (expenses > 0), False, df.CryoSleep)

    return df


def fill_missing_age(df, expenses):
    """
    Fill missing Age values based on expenses conditions.
    """
    median_without_expenses = df[(df['Age'] < 13)]['Age'].median()
    median_with_expenses = df[(df['Age'] > 12)]['Age'].median()
    df['Age'] = np.where(df['Age'].isnull() & (expenses == 0), median_without_expenses, df['Age'])
    df['Age'] = np.where(df['Age'].isnull() & (expenses > 0), median_with_expenses, df['Age'])

    return df


def fill_missing_deckno(df):
    """
    Fill missing DeckNo values based on HomePlanet conditions.
    """
    conditions = [
        (df["DeckNo"].isnull() & df.HomePlanet.str.contains('Earth')),
        (df["DeckNo"].isnull() & df.HomePlanet.str.contains('Europa')),
        (df["DeckNo"].isnull() & df.HomePlanet.str.contains('Mars'))
    ]
    values = ['G', 'B', 'F']

    df['DeckNo'] = np.select(conditions, values, default=df["DeckNo"])

    return df


def fill_missing_deckside(df):
    """
    Fill missing DeckSide values based on LastName conditions.
    """
    df['DeckSize'] = np.where(df["DeckSize"].isnull() & df['LastName'].eq(df['LastName'].shift()),
                              df["DeckSize"].shift(), df["DeckSize"])
    return df


def fill_missing_values(df, expenses):
    """
    Fill missing values in the DataFrame based on specific conditions.
    Args:
    df : DataFrame : input data
    expenses : Series : input data for expenses
    """
    df = fill_missing_vip(df)
    df = fill_missing_homeplanet(df)
    df = fill_missing_cryosleep(df, expenses)
    df = fill_missing_age(df, expenses)
    df = fill_missing_deckno(df)
    df = fill_missing_deckside(df)

    return df


def preprocess(df, target_column):
    # 3. Feature extraction
    passenger_ids = df.loc[df['SetId'] == 'Test', 'PassengerId'].values
    df = split_column(df, 'PassengerId', '_', ['GroupId', 'SubgroupId'])
    df = split_column(df, 'Cabin', '/', ['DeckNo', 'DeckNum', 'DeckSize'])
    df = split_column(df, 'Name', ' ', ['FirstName', 'LastName'])

    # df = get_family_id(df)

    analyze_data(df)

    missing_values = df.groupby('SetId').apply(lambda x: x.isna().sum())
    print(missing_values)

    # df['Transported'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()

    EXPENSES_COLUMNS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    expenses = 0
    for ex in EXPENSES_COLUMNS:
        expenses += df[ex]
    df = fill_missing_values(df, expenses)
    # df['DeckNum'] = np.where(df.DeckNum.isnull() & df.LastName.eq(df.LastName.shift()),
    #                      df.DeckNum.shift(), df.DeckNum)

    df.loc[((df['ShoppingMall'] > 0) |
            (df['FoodCourt'] > 0) |
            (df['Spa'] > 0) |
            (df['VRDeck'] > 0) |
            (df['RoomService'] > 0)) &
           df['CryoSleep'].isna(), 'CryoSleep'] = False
    df.loc[(df['CryoSleep'] == True) &
           ((df['ShoppingMall'].isna()) |
            (df['FoodCourt'].isna()) |
            (df['Spa'].isna()) |
            (df['VRDeck'].isna()) |
            (df['RoomService'].isna())), ['ShoppingMall', 'FoodCourt', 'Spa', 'VRDeck', 'RoomService']] = 0

    # df.isna().sum().plot(kind='bar')
    # plt.show()

    print(df.info())
    print(df.dtypes)

    # 5. Encoding of categorical variables
    # One hot encoding
    cols_one_hot_encoded = ['HomePlanet', 'Destination']
    df = one_hot_encoding(df, cols_one_hot_encoded)
    # Ordinal encoding
    oe = OrdinalEncoder()
    cols_ordinal_encoded = ['CryoSleep', 'VIP', 'DeckSize', 'Transported', 'DeckNo','DeckNum']
    df[cols_ordinal_encoded] = oe.fit_transform(df[cols_ordinal_encoded])

    # 4. Filling missing values
    categorical_var_names, numerical_var_names = get_column_types(df)
    new_column_name = pd.Index([numerical_var_names])
    all_column_names = categorical_var_names.append(new_column_name)

    df = num_inputation(df=df)

    missing_values = df.groupby('SetId').apply(lambda x: x.isna().sum())
    print(missing_values)

    X_test = df.loc[df['SetId'] == 'Test']
    X = df.loc[df['SetId'] == 'Train']
    y = X[target_column].values.astype(int)

    selected_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    X.drop(columns=['SetId', 'FirstName', 'LastName', 'DeckNum'], axis=1, inplace=True)
    X_test.drop(columns=['SetId', target_column, 'FirstName', 'LastName', 'DeckNum'], axis=1, inplace=True)

    # X.drop(columns=['SetId', 'FirstName', 'LastName'], axis=1, inplace=True)
    # X_test.drop(columns=['SetId', target_column, 'FirstName', 'LastName'], axis=1, inplace=True)

    X = X.drop(target_column, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train[selected_columns] = scaler.fit_transform(X_train[selected_columns])
    X_val[selected_columns] = scaler.transform(X_val[selected_columns])
    X_test[selected_columns] = scaler.fit_transform(X_test[selected_columns])

    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    # Modelling
    models = [
        ('Random Forest', RandomForestClassifier()),
        ('XGBoost', XGBClassifier()),
        ('LightGBM', LGBMClassifier()),
        ('CatBoost', CatBoostClassifier(silent=True)),
    ]
    results = []
    for model_name, model in models:
        y_train_pred, y_val_pred = fit_and_predict(model, X_train, y_train, X_val)
        result = print_metrics(model_name, y_train, y_train_pred, y_val, y_val_pred)
        results.append(result)

    results_df = pd.DataFrame(results)
    print(results_df)

    return X_train, X_val, y_train, y_val, X_test, passenger_ids
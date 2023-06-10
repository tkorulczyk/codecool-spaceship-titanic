import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, \
    matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.model_selection import cross_validate


def perform_cross_validation(model, X, y, scoring):
    skf = StratifiedKFold(n_splits=10)
    return cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=True)


def calculate_averaged_scores(scores):
    avg_scores = {}
    for metric, values in scores.items():
        avg_scores[metric] = round(sum(values) / len(values), 3)
        avg_scores[f"{metric}_std"] = round(np.std(values), 3)  # Include standard deviation
    return avg_scores


def compute_confusion_matrix_diff(y_train, y_val, y_pred_train, y_pred_val):
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_val = confusion_matrix(y_val, y_pred_val)

    TP_train, FP_train, FN_train, TN_train = cm_train.ravel()
    TP_val, FP_val, FN_val, TN_val = cm_val.ravel()

    total_train = TP_train + FP_train + FN_train + TN_train
    total_val = TP_val + FP_val + FN_val + TN_val

    TP_diff = (TP_val / total_val - TP_train / total_train)
    TN_diff = (TN_val / total_val - TN_train / total_train)
    FP_diff = (FP_val / total_val - FP_train / total_train)
    FN_diff = (FN_val / total_val - FN_train / total_train)

    return np.array([TP_diff, TN_diff, FP_diff, FN_diff])


def format_results(cross_val_results_df, scoring):
    train_data = {f"{col}_M": [f"{row['train_' + col]:.3f} [{row['train_' + col + '_std']:.3f}]" for _, row in
                               cross_val_results_df.iterrows()] for col in scoring}
    train_df = pd.DataFrame(train_data)

    val_data = {f"{col}_M": [f"{row['test_' + col]:.3f} [{row['test_' + col + '_std']:.3f}]" for _, row in
                             cross_val_results_df.iterrows()] for col in scoring}
    val_df = pd.DataFrame(val_data)

    train_df['Model'] = cross_val_results_df['Model']
    val_df['Model'] = cross_val_results_df['Model']

    train_df.set_index('Model', inplace=True)
    val_df.set_index('Model', inplace=True)

    new_train_columns = {col: col.replace("_M", "").capitalize().ljust(15) for col in train_df.columns}
    new_val_columns = {col: col.replace("_M", "").capitalize().ljust(15) for col in val_df.columns}

    train_df.rename(columns=new_train_columns, inplace=True)
    val_df.rename(columns=new_val_columns, inplace=True)

    def format_number(number):
        if isinstance(number, (int, float)):
            return f"{number:.3f}"
        return number

    train_df = train_df.applymap(format_number)
    val_df = val_df.applymap(format_number)

    return train_df, val_df


def process_and_print_results(cross_val_results, scoring):
    cross_val_results_df = pd.DataFrame(cross_val_results)

    diff_df = cross_val_results_df[['Model', 'TP_diff (%)', 'TN_diff (%)', 'FP_diff (%)', 'FN_diff (%)']].copy()

    train_df, val_df = format_results(cross_val_results_df, scoring)

    print("Results for Train set [Metric, Mean, SD]:")
    print(train_df.round(3))

    print("\nResults for Validation set [Metric, Mean, SD]:")
    print(val_df.round(3))

    print("\nDifferences in Confusion Matrix Elements (%):")
    print(diff_df.round(3))


def print_crossvalidated_metrics(models, scoring, X_train, y_train, X_val, y_val, X, y, show_confusion_matrices=False):
    cross_val_results = []

    for model_name, model in models:
        # Perform cross-validation
        scores = perform_cross_validation(model, X, y, scoring)
        avg_scores = calculate_averaged_scores(scores)

        if show_confusion_matrices:
            show_cros_validated_confusion_matrices(X, y, model, model_name)

        y_pred_train = cross_val_predict(model, X_train, y_train, cv=10)
        y_pred_val = cross_val_predict(model, X_val, y_val, cv=10)

        diff = compute_confusion_matrix_diff(y_train, y_val, y_pred_train, y_pred_val)

        cross_val_results.append(
            {'Model': model_name, **avg_scores, 'TP_diff (%)': diff[0], 'TN_diff (%)': diff[1], 'FP_diff (%)': diff[2],
             'FN_diff (%)': diff[3]})

    # Processing and printing results
    process_and_print_results(cross_val_results, scoring)


def show_cros_validated_confusion_matrices(X, y, model, model_name):
    """
    Displays confusion matrix heatmaps for training and validation datasets side by side using cross-validation.

    Parameters:
    - X: Features
    - y: Labels
    - model: Classifier
    """

    # Definiowanie walidacji krzyżowej
    kf = KFold(n_splits=10)
    skf = StratifiedKFold(n_splits=10)

    # Macierz, która przechowuje sumę wszystkich macierzy pomyłek
    sum_cm_train = np.zeros((2, 2))
    sum_cm_val = np.zeros((2, 2))

    # Iteracja przez podziały walidacji krzyżowej
    for train_index, val_index in skf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Trenowanie modelu na podzbiorze treningowym
        model.fit(X_train, y_train)

        # Prognozowanie na podzbiorze treningowym i walidacyjnym
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Obliczanie macierzy pomyłek dla tego podziału
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_val = confusion_matrix(y_val, y_pred_val)

        # Dodawanie tej macierzy pomyłek do sumy
        sum_cm_train += cm_train
        sum_cm_val += cm_val

    # Uśrednianie macierzy pomyłek
    average_cm_train = sum_cm_train / skf.get_n_splits()
    average_cm_val = sum_cm_val / skf.get_n_splits()

    # Define the custom color map
    custom_colormap = sns.color_palette(["#FFFFFF", "#B51F20"])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Function to format the annotations
    def fmt(x, total):
        pct = int(round(100 * x / total))
        return f'[{pct}%] {int(x)}'

    total_train = np.sum(average_cm_train)
    total_val = np.sum(average_cm_val)

    # Heatmap for training dataset
    sns.heatmap(average_cm_train, annot=np.vectorize(fmt)(average_cm_train, total_train), fmt='', cmap=custom_colormap,
                cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['True Positive', 'True Negative'], ax=ax1)
    ax1.set_title(f'Average Confusion Matrix (Training Data) - {model_name}')
    ax1.set_xlabel('Predicted Classes')
    ax1.set_ylabel('True Classes')
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('#B51F20')
        spine.set_linewidth(2)

    # Heatmap for validation dataset
    sns.heatmap(average_cm_val, annot=np.vectorize(fmt)(average_cm_val, total_val), fmt='', cmap=custom_colormap,
                cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['True Positive', 'True Negative'], ax=ax2)
    ax2.set_title(f'Average Confusion Matrix (Validation Data) - {model_name}')
    ax2.set_xlabel('Predicted Classes')
    ax2.set_ylabel('True Classes')
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        spine.set_color('#B51F20')
        spine.set_linewidth(2)

    # Display the plots
    plt.show()


def show_confusion_matrices(y_true_train, y_pred_train, y_true_val, y_pred_val):
    """
    Displays confusion matrix heatmaps for training and validation datasets side by side.

    Parameters:
    - y_true_train: Ground truth (correct) target values for training dataset.
    - y_pred_train: Estimated targets as returned by a classifier for training dataset.
    - y_true_val: Ground truth (correct) target values for validation dataset.
    - y_pred_val: Estimated targets as returned by a classifier for validation dataset.
    """

    # Calculate confusion matrices
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_val = confusion_matrix(y_true_val, y_pred_val)

    # Define the custom color map
    custom_colormap = sns.color_palette(["#FFFFFF", "#B51F20"])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calculate the mask to highlight FP and FN
    mask_train = np.array([[0, 1], [1, 0]]) * cm_train
    max_val_train = max(mask_train.flatten())

    mask_val = np.array([[0, 1], [1, 0]]) * cm_val
    max_val_val = max(mask_val.flatten())

    # Function to format the annotations
    def fmt(x, total):
        pct = int(100 * x / total)
        return f'[{pct}%] {x}'

    total_train = np.sum(cm_train)
    total_val = np.sum(cm_val)

    # Heatmap for training dataset
    sns.heatmap(mask_train, annot=np.vectorize(fmt)(cm_train, total_train), fmt='', cmap=custom_colormap, cbar=False,
                vmax=max_val_train,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['True Positive', 'True Negative'], ax=ax1)
    ax1.set_title('Confusion Matrix (Training Data)')
    ax1.set_xlabel('Predicted Classes')
    ax1.set_ylabel('True Classes')
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('#B51F20')
        spine.set_linewidth(2)

    # Heatmap for validation dataset
    sns.heatmap(mask_val, annot=np.vectorize(fmt)(cm_val, total_val), fmt='', cmap=custom_colormap, cbar=False,
                vmax=max_val_val,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['True Positive', 'True Negative'], ax=ax2)
    ax2.set_title('Confusion Matrix (Validation Data)')
    ax2.set_xlabel('Predicted Classes')
    ax2.set_ylabel('True Classes')
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        spine.set_color('#B51F20')
        spine.set_linewidth(2)

    # Display the plots
    plt.show()


def compute_metrics(y, ŷ, avg='weighted'):
    """
    Calculate and return various evaluation metrics for given ground truth labels and predicted labels.

    Parameters:
    y (array-like): Ground truth labels.
    ŷ (array-like): Predicted labels.
    average (str, optional): The averaging method to be used for calculating precision, recall, and F1-score.
                              Defaults to 'weighted'.

    Returns:
    tuple: A tuple containing the following metrics (in order): accuracy, precision, recall, F1-score,
           Jaccard score, ROC AUC score, and Matthews correlation coefficient.
    """

    def get_metric(func, average):
        return lambda y, ŷ: func(y, ŷ, average=average)

    metrics_funcs = [
        accuracy_score,
        get_metric(precision_score, avg),
        get_metric(recall_score, avg),
        get_metric(f1_score, avg),
        get_metric(jaccard_score, avg),
        roc_auc_score,
        matthews_corrcoef
    ]

    return tuple(round(func(y, ŷ), 3) for func in metrics_funcs)


def print_metrics(model_name, **kwargs):
    """
    Calculate and return various evaluation metrics for classification models. Below are the descriptions and definitions of each metric:

    1. **Accuracy**: Accuracy is the ratio of the number of correctly classified instances to the total number of instances. It is defined as (True Positives + True Negatives) / (True Positives + False Positives + True Negatives + False Negatives). It tells how often the classifier is correct.
    2. **Precision**: Precision is the ratio of the number of true positive instances to the number of instances classified as positive. It is defined as True Positives / (True Positives + False Positives). It tells how often the positive predictions are correct.
    3. **Recall or Sensitivity**: Recall is the ratio of the number of true positive instances to the number of actual positive instances. It is defined as True Positives / (True Positives + False Negatives). It tells how often the classifier correctly identifies positive instances.
    4. **F1 Score**: The F1 Score is the harmonic mean of precision and recall. It tries to find the balance between precision and recall. It is defined as 2 * (Precision * Recall) / (Precision + Recall).
    5. **Jaccard Index**: The Jaccard Index, or the Jaccard similarity coefficient, is a statistic used for comparing the similarity and diversity of sample sets. For binary classification, it is defined as True Positives / (True Positives + False Positives + False Negatives).
    6. **ROC AUC**: Receiver Operating Characteristic Area Under the Curve (ROC AUC) is a performance measurement for classification problems. It tells how much the model is capable of distinguishing between classes. The higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s.
    7. **Matthews Correlation Coefficient (MCC)**: MCC is a metric used in machine learning as a measure of the quality of binary classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure that can be used even if the classes are of very different sizes. It is defined as (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)).

    Parameters:
    - model_name (str): The name of the classification model.
    - kwargs: Variable length keyword arguments representing datasets.

    Returns:
    - dict: A dictionary with the evaluation metrics.
    """

    # Names of the metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Jaccard', 'ROC AUC', 'Matthews Corr']

    # Inner function to compute the metrics
    def compute_metric(y, ŷ):
        """
        Calculate evaluation metrics for classification.

        Parameters:
        - y (array-like): The ground truth labels.
        - ŷ (array-like): The predicted labels.

        Returns:
        - dict: A dictionary with the evaluation metrics.
        """

        metrics_values = compute_metrics(y, ŷ)
        return {name: value for name, value in zip(metrics_names, metrics_values)}

    # Results dictionary to hold the metrics
    results = {}

    # Loop through the datasets
    for suffix, (y, ŷ) in kwargs.items():
        # Compute the metrics for this dataset
        metrics_result = compute_metric(y, ŷ)

        # Compute the number of mislabeled instances
        mislabeled = (y != ŷ).sum()

        # Update the results dictionary
        results.update({'Model': model_name, **{f"{name} {suffix}": value for name, value in metrics_result.items()},
                        f'Mislabeled {suffix}': mislabeled})

    # Return the results dictionary
    return results

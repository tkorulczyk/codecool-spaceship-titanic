# Spaceship Titanic Kaggle Challenge

## Task description from Kaggle
Welcome to the Spaceship Titanic Dimensional Anomaly Challenge! The year is 2912 and we have encountered an interstellar catastrophe. The Spaceship Titanic, carrying approximately 13,000 passengers, collided with a spacetime anomaly while en route to 55 Cancri E. Nearly half of the passengers have been transported to an alternate dimension! You must use your data science skills to help rescue the lost passengers by predicting which ones were transported.

## Dataset

You will be working with a dataset that consists of personal records of passengers aboard the Spaceship Titanic. The dataset is divided into two parts: a training set and a test set.

## File Descriptions

train.csv - Training set with personal records for about two-thirds (~8700) of the passengers. This file contains the target column Transported, which indicates whether the passenger was transported to another dimension.

test.csv - Test set with personal records for the remaining one-third (~4300) of the passengers. You must predict the value of Transported for each passenger in this set.

ssample_submission.csv - A sample submission file in the correct format.

## Variables description

- **PassengerId** - A unique Id for each passenger, where the form is `gggg_pp` (gggg is the group, and pp is the number within the group).
- **HomePlanet** - The planet from which the passenger departed.
- **CryoSleep** - Whether the passenger elected to be put into suspended animation for the duration of the voyage.
- **Cabin** - The cabin number where the passenger is staying (format `deck/num/side`, where side is P for Port or S for Starboard).
- **Destination** - The planet to which the passenger will be disembarking.
- **Age** - The age of the passenger.
- **VIP** - Whether the passenger has paid for special VIP service during the voyage.
- **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck** - Amount billed at each of the Spaceship Titanic's amenities.
- **Name** - The first and last names of the passenger.
- **Transported** - Whether the passenger was transported to another dimension (target variable).
- 
## Variables characteristics after initial transformation
Data columns (total 21 columns):
 | Index | Column Name  | Non-Null Count | Data Type |
|-------|--------------|----------------|-----------|
| 0     | HomePlanet   | 12967          | object    |
| 1     | CryoSleep    | 12957          | object    |
| 2     | Destination  | 12696          | object    |
| 3     | Age          | 12947          | category  |
| 4     | VIP          | 12926          | object    |
| 5     | RoomService  | 12800          | float64   |
| 6     | FoodCourt    | 12790          | float64   |
| 7     | ShoppingMall | 12795          | float64   |
| 8     | Spa          | 12793          | float64   |
| 9     | VRDeck       | 12793          | float64   |
| 10    | Transported  | 8693           | object    |
| 11    | SetId        | 12970          | object    |
| 12    | GroupId      | 12970          | object    |
| 13    | SubgroupId   | 12970          | object    |
| 14    | DeckNo       | 12970          | object    |
| 15    | DeckNum      | 12671          | object    |
| 16    | DeckSize     | 12739          | object    |
| 17    | FirstName    | 12676          | object    |
| 18    | LastName     | 12676          | object    |
| 19    | FamilySize   | 12676          | float64   |
| 20    | FamilyId     | 12970          | object    |

dtypes: category(1), float64(6), object(14)

## Characteristics of missing values

| Column Name  | Test | Train |
|--------------|------|-------|
| HomePlanet   | 87   | 201   |
| CryoSleep    | 93   | 217   |
| Destination  | 92   | 182   |
| Age          | 91   | 179   |
| VIP          | 93   | 203   |
| RoomService  | 82   | 181   |
| FoodCourt    | 106  | 183   |
| ShoppingMall | 98   | 208   |
| Spa          | 101  | 183   |
| VRDeck       | 80   | 188   |
| Transported  | 4277 | 0     |
| SetId        | 0    | 0     |
| GroupId      | 0    | 0     |
| SubgroupId   | 0    | 0     |
| DeckNo       | 100  | 199   |
| DeckNum      | 100  | 199   |
| DeckSize     | 100  | 199   |
| FirstName    | 94   | 200   |
| LastName     | 94   | 200   |
| FamilySize   | 94   | 200   |
| FamilyId     | 0    | 0     |

# Overview of the Code

The provided code is structured in two python files: `main.py` and `tunner.py`. The `main.py` file is the entry point, where the model training, hyperparameter tuning, and prediction are performed. The `tunner.py` contains a utility function for hyperparameter tuning. The program uses several libraries such as pandas, optuna, Keras, LightGBM, XGBoost, and CatBoost for various tasks such as data preprocessing, hyperparameter tuning, and model training.

The workflow of the code is as follows:

1. **Data Import**: Load training and test datasets.
2. **Data Preprocessing**: Combine the datasets, clean, and preprocess them.
3. **Hyperparameter Tuning**: Utilize Optuna to tune hyperparameters for CatBoost, LightGBM, and XGBoost models.
4. **Model Training**: Train the selected model using the best hyperparameters.
5. **Prediction and Submission**: Make predictions on the test set and submit them to Kaggle (optional).

# Detailed Explanation

## main.py

### Importing Libraries
The script begins by importing necessary libraries such as NumPy, Optuna, Pandas, Keras, LightGBM, XGBoost, and CatBoost. These libraries are essential for data manipulation, hyperparameter tuning, and model building.

### Configuration
The script then sets various configurations such as disabling warnings, setting Pandas display options, and configuring Optuna's logging.

### Flags and Modes
There are three running modes set by the flag `running_mode`:
   - 'tunner': Performs hyperparameter tuning using Optuna.
   - 'nn': Trains a neural network using Keras.
   - 'standard': Trains gradient boosting models with predefined or tuned hyperparameters.

### Data Import
Data is imported from CSV files into Pandas DataFrames. Additional columns `SetId` are added to keep track of training and test sets.

### Preprocessing
The `preprocess` function is called from a locally imported module, which takes care of preprocessing the data including handling missing values, encoding categorical features, and splitting the data into training, validation, and test sets.

### Hyperparameter Tuning
When `running_mode` is set to 'tunner', the script initializes an Optuna study object and performs hyperparameter tuning for CatBoost, LightGBM, and XGBoost models. The `objective` function from the `tunner.py` file is used here.

### Neural Network Training
When `running_mode` is set to 'nn', the script builds and trains a neural network using Keras. The neural network architecture includes Embedding layers for categorical features and Dense layers for numerical features.

### Gradient Boosting Training
When `running_mode` is set to 'standard', gradient boosting models (CatBoost, LightGBM, and XGBoost) are trained with either predefined or tuned hyperparameters. Cross-validated metrics are printed.

### Prediction and Submission
The script predicts the target variable for the test dataset using the trained model. Optionally, if `submit_to_kaggle` flag is set to True, the predictions are saved to a CSV file and submitted to a Kaggle competition using a subprocess command.

## tunner.py

### train_and_evaluate_model Function
This function accepts the best parameters from the tuning process, trains a CatBoost model, and returns predictions for the training and validation sets.

### objective Function

This function is responsible for hyperparameter tuning. It accepts an Optuna `trial` object, training data, and a `model_type` indicating which model to tune ('catboost', 'xgboost', or 'lightgbm'). It defines a search space for hyperparameters and trains the model using different sets of hyperparameters. The performance is evaluated on a validation set, and Optuna keeps track of the best hyperparameters that yield the best performance.

## preprocess.py

## Libraries and Dependencies

- `numpy` : For numerical operations.
- `pandas` : For data manipulation and analysis.
- `sklearn.experimental` : To enable experimental features.
- `sklearn.impute` : For imputing missing values.
- `sklearn.model_selection` : For splitting dataset.
- `sklearn.preprocessing` : For encoding categorical features and standardization.
- `tabulate` : For pretty printing tabular data.

## Functions

### 1. get_FamilyId(df)

This function creates a new feature called `FamilyId` that groups individuals based on their `GroupId` and `LastName`.

### 2. one_hot_encoding(df, columns)

Encodes categorical variables using one-hot encoding.

- `df` : The DataFrame to encode.
- `columns` : The columns to be one-hot encoded.

### 3. num_inputation(df, n_iterations=10)

Imputes missing values in numerical features using an iterative imputer.

- `df` : The DataFrame to impute.
- `n_iterations` : The number of imputing iterations.

### 4. split_column(df, column_name, separator, new_column_names)

Splits a column into multiple columns based on a separator.

- `df` : The DataFrame to manipulate.
- `column_name` : The name of the column to split.
- `separator` : The separator to use for the split.
- `new_column_names` : The names of the new columns.

### 5. analyze_data(df)

Analyzes data and prints summaries of categorical and numerical variables.

- `df` : The DataFrame to analyze.

### 6. get_column_types(df_train, acceptable_duplication_threshold=0.05)

Identifies column types as categorical or numerical.

- `df_train` : The training DataFrame.
- `acceptable_duplication_threshold` : Threshold for considering a column as categorical.

### 7. fill_missing_vip(df)

Fills in missing VIP values based on age and HomePlanet.

- `df` : The DataFrame.

### 8. fill_missing_homeplanet(df)

Fills in missing HomePlanet values based on DeckNo and VIP.

- `df` : The DataFrame.

### 9. fill_missing_cryosleep(df, expenses)

Fills in missing CryoSleep values based on expenses.

- `df` : The DataFrame.
- `expenses` : Expenses data.

### 10. fill_missing_age(df, expenses)

Fills in missing Age values based on expenses.

- `df` : The DataFrame.
- `expenses` : Expenses data.

### 11. fill_missing_deckno(df)

Fills in missing DeckNo values based on HomePlanet.

- `df` : The DataFrame.

### 12. fill_missing_deckside(df)

Fills in missing DeckSide values based on LastName.

- `df` : The DataFrame.

### 13. fill_missing_values(df)

Fills in all missing values using various conditions.

- `df` : The DataFrame.

### 14. group_age(df)

Groups ages into bins.

- `df` : The DataFrame.

### 15. preprocess(df, target_column)

Preprocesses the data.

- `df` : The DataFrame.
- `target_column` : The target variable.

## Workflow

1. Feature extraction and engineering.
2. Handling missing values.
3. Encoding categorical variables.
4. Standardization of features.
5. Train-test splitting.

## Output

Returns preprocessed data:

- Training data.
- Validation data.


# Conclusion

This script **automates** the machine learning process for building and tuning models. It's well-organized, modular, and could be a strong starting point for anyone participating in **Kaggle competitions** or working on machine learning projects. However, one might need to modify the preprocessing steps and model architecture to suit the specific dataset and problem at hand.

Key points in the script include data preprocessing, hyperparameter tuning using **Optuna**, training models (including **gradient boosting models** and **neural networks**), and optionally submitting predictions to Kaggle.
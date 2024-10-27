import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def clean_data(train, test):
    # Inspect data structure (optional)
    pd.set_option('display.max_columns', None)
    print(f"Train Dataset Structure: \n {train.head()}")
    print(f"Test Dataset Structure: \n {test.head()}")

    # Drop irrelevant columns
    y_train = train['Survived']
    x_train = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
    x_test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    # Optionally process Cabin to extract deck letter (e.g., C, D, etc.)
    for dataset in [x_train, x_test]:
        dataset['Deck'] = dataset['Cabin'].str[0]  # Extract the first letter
        dataset.drop('Cabin', axis=1, inplace=True)  # Drop original Cabin column

    # Handle missing values: Fill Age with median and drop rows with missing Embarked
    x_train.loc[:, 'Age'] = x_train['Age'].fillna(x_train['Age'].median())
    x_test.loc[:, 'Age'] = x_test['Age'].fillna(x_test['Age'].median())

    # Fill missing values for Embarked in test data with mode of training data
    x_test.loc[:, 'Embarked'] = x_test['Embarked'].fillna(x_train['Embarked'].mode()[0])

    return x_train, x_test, y_train

def process_data(train_cleaned, test_cleaned):
    x_train, x_test = train_cleaned, test_cleaned

    # Binary encode Sex
    le_sex = LabelEncoder()
    x_train['Sex'] = le_sex.fit_transform(x_train['Sex'])
    x_test['Sex'] = le_sex.transform(x_test['Sex'])

    # One-hot encode Embarked and Deck
    x_train = pd.get_dummies(x_train, columns=['Embarked', 'Deck'], drop_first=False)
    x_test = pd.get_dummies(x_test, columns=['Embarked', 'Deck'], drop_first=False)

    # Align columns in case of mismatch
    x_train, x_test = x_train.align(x_test, join='outer', axis=1, fill_value=0)

    # Normalize Age and Fare
    scaler = StandardScaler()
    x_train[['Age', 'Fare']] = scaler.fit_transform(x_train[['Age', 'Fare']])
    x_test[['Age', 'Fare']] = scaler.transform(x_test[['Age', 'Fare']])

    return x_train, x_test

def create_data_loaders(train_processed, test_processed, y_train, batch_size=32):
    # Ensure all data is numeric and handle NaN values
    train_processed = train_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_processed = test_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)  # Handle NaNs in labels as well

    # Convert boolean columns to integers (0 or 1)
    train_processed = train_processed.astype(int)
    test_processed = test_processed.astype(int)

    # Create tensors
    train_tensor = torch.tensor(train_processed.values).float()  # Convert to float tensor
    y_train_tensor = torch.tensor(y_train.values).float()  # Labels as tensor
    test_tensor = torch.tensor(test_processed.values).float()

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    test_dataset = TensorDataset(test_tensor)  # Test dataset does not have labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
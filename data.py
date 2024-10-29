import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def clean_data(train, test):
    # Extract target variable
    y_train = train['Survived']

    # Extract Title from Name
    for dataset in [train, test]:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Family Size feature
    for dataset in [train, test]:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
        dataset['IsAlone'] = (dataset['FamilySize'] == 0).astype(int)

    # Deck feature extraction
    for dataset in [train, test]:
        dataset['Deck'] = dataset['Cabin'].str[0]  # Extract first letter as Deck level

    # Fill missing values
    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)
    for dataset in [train, test]:
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # Drop irrelevant columns
    x_train = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    x_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return x_train, x_test, y_train


def process_data(train_cleaned, test_cleaned):
    x_train, x_test = train_cleaned, test_cleaned

    # Binary encode Sex
    le_sex = LabelEncoder()
    x_train['Sex'] = le_sex.fit_transform(x_train['Sex'])
    x_test['Sex'] = le_sex.transform(x_test['Sex'])

    # One-hot encode categorical features (Embarked, Title, Deck)
    x_train = pd.get_dummies(x_train, columns=['Embarked', 'Title', 'Deck'], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=['Embarked', 'Title', 'Deck'], drop_first=True)

    # Ensure train and test datasets have the same columns after one-hot encoding
    x_train, x_test = x_train.align(x_test, join='outer', axis=1, fill_value=0)

    # Feature binning for Age and Fare
    x_train['AgeBin'] = pd.cut(x_train['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    x_test['AgeBin'] = pd.cut(x_test['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    x_train['FareBin'] = pd.qcut(x_train['Fare'], 4, labels=False)
    x_test['FareBin'] = pd.qcut(x_test['Fare'], 4, labels=False)

    # Interaction features
    x_train['ClassFare'] = x_train['Pclass'] * x_train['Fare']
    x_test['ClassFare'] = x_test['Pclass'] * x_test['Fare']
    x_train['AgeClass'] = x_train['Age'] * x_train['Pclass']
    x_test['AgeClass'] = x_test['Age'] * x_test['Pclass']

    # Polynomial features for Age and Fare
    x_train['Age^2'] = x_train['Age'] ** 2
    x_train['Fare^2'] = x_train['Fare'] ** 2
    x_test['Age^2'] = x_test['Age'] ** 2
    x_test['Fare^2'] = x_test['Fare'] ** 2

    # Normalize continuous features
    scaler = StandardScaler()
    continuous_features = ['Age', 'Fare', 'FamilySize', 'ClassFare', 'AgeClass', 'Age^2', 'Fare^2']
    x_train[continuous_features] = scaler.fit_transform(x_train[continuous_features])
    x_test[continuous_features] = scaler.transform(x_test[continuous_features])

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
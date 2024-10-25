import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def clean_data(train, test):
    # Inspecting Datasets Structures
    pd.set_option('display.max_columns', None)
    print(f"Train Dataset Structure: \n {train.head()}")
    print(f"Test Dataset Structure: \n {test.head()}")

    # Dropping Unnecessary Columns (PassengerID, Survived, Name, Cabin, Ticket)
    y_train = train['Survived']
    x_train = train.drop(['PassengerId', 'Survived', 'Name', 'Cabin', 'Ticket'], axis=1)
    x_test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

    # Inspecting Modified Datasets
    print(f"Y_Train Structure: \n {y_train.head()}")
    print(f"X_Train Structure: \n {x_train.head()}")
    print(f"X_Test Structure: \n {x_test.head()}")

    return x_train, x_test, y_train

def process_data(train_cleaned, test_cleaned):
    x_train = train_cleaned
    x_test = test_cleaned

    # Convert 'Sex' to binary values
    le_sex = LabelEncoder()
    x_train['Sex'] = le_sex.fit_transform(x_train['Sex'])
    x_test['Sex'] = le_sex.transform(x_test['Sex'])

    # Convert 'Embarked' to binary values with one-hot encoding, keeping all categories
    x_train = pd.get_dummies(x_train, columns=['Embarked'], drop_first=False)  # Set drop_first=False
    x_test = pd.get_dummies(x_test, columns=['Embarked'], drop_first=False)    # Set drop_first=False

    # Align columns in case they differ after encoding
    # Use 'outer' to include all categories
    x_train, x_test = x_train.align(x_test, join='outer', axis=1, fill_value=0)

    # Normalize 'Age' and 'Fare' columns
    scaler = StandardScaler()
    x_train[['Age', 'Fare']] = scaler.fit_transform(x_train[['Age', 'Fare']])
    x_test[['Age', 'Fare']] = scaler.transform(x_test[['Age', 'Fare']])

    # Inspecting Modified Datasets
    print(f"X_Train Structure after encoding and normalization: \n{x_train.head()}")
    print(f"X_Test Structure after encoding and normalization: \n{x_test.head()}")

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
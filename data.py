import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# Data cleaning and simplified feature engineering function
def clean_and_process_data(train, test):
    # Define target variable
    y_train = train['Survived']

    # Simplify feature engineering
    for dataset in [train, test]:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
        dataset['IsAlone'] = (dataset['FamilySize'] == 0).astype(int)

    # Drop unnecessary columns
    train = train[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']]
    test = test[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']]

    # Handle missing values
    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)
    for dataset in [train, test]:
        dataset['Embarked'].fillna('S', inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

    # Standardize continuous features
    scaler = StandardScaler()
    train[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(train[['Age', 'Fare', 'FamilySize']])
    test[['Age', 'Fare', 'FamilySize']] = scaler.transform(test[['Age', 'Fare', 'FamilySize']])

    return train, test, y_train


# Prepare data loaders
def create_data_loaders(train, y_train, test, batch_size=32):
    train_tensor = torch.tensor(train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    test_tensor = torch.tensor(test.values, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
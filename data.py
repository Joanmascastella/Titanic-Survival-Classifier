import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# Data cleaning and simplified feature engineering function
def clean_and_process_data(train, test):
    # Define target variable
    y_train = train['Survived'].copy()

    # Simplify feature engineering
    for dataset in [train, test]:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
        dataset['IsAlone'] = (dataset['FamilySize'] == 0).astype(int)

    # Select and drop unnecessary columns, avoiding inplace assignments
    train = train[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']].copy()
    test = test[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']].copy()

    # Handle missing values with `.loc` instead of `inplace=True`
    train['Age'] = train['Age'].fillna(train['Age'].median())
    test['Age'] = test['Age'].fillna(test['Age'].median())
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())
    for dataset in [train, test]:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Encode categorical features
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

    # Standardize continuous features with `.loc`
    scaler = StandardScaler()
    continuous_features = ['Age', 'Fare', 'FamilySize']
    train[continuous_features] = scaler.fit_transform(train[continuous_features].values)
    test[continuous_features] = scaler.transform(test[continuous_features].values)

    return train, test, y_train


# Prepare data loaders with a train-validation split
def create_data_loaders(train, y_train, batch_size=32, test_size=0.2, random_state=42):
    # Perform train-validation split
    train_data, val_data, y_train_data, y_val_data = train_test_split(
        train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )

    # Convert data to tensors
    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_data.values, dtype=torch.float32).unsqueeze(1)
    val_tensor = torch.tensor(val_data.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_data.values, dtype=torch.float32).unsqueeze(1)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    val_dataset = TensorDataset(val_tensor, y_val_tensor)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

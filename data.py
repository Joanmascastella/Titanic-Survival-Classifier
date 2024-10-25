import pandas as pd



def clean_data(train, test):
    # Inspecting Datasets Structures
    pd.set_option('display.max_columns', None)
    print(f"Train Dataset Structure: \n {train.head()}")
    print(f"Test Dataset Structure: \n {test.head()}")

    # Dropping Unnecessary Columns (PassengerID, Name, Ticket)


    # Convert Survival, PClass, Sex, Embarked To Binary Values


    return train, test


def process_data(train_cleaned, test_cleaned):

    return train_cleaned, test_cleaned

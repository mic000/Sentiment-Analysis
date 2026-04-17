import pandas as pd
import os, re, csv

from sklearn.model_selection import train_test_split


def clean_text (file_path, name_file, char = None):
    """
    stage 1: clean raw data

    :param file_path: the pathway to find folder which store raw data and reside filtered dataset
    :param name_file: name of the data file
    :param char: keep number of characters in each sentence or row

    :return: write the filtered data file and store at the parent folder as CSV file
    """

    # set raw data input and filtered data output pathway
    # store filtered datasets into folders
    input_path = os.path.join(file_path, name_file + ".txt")
    output_path = os.path.join(os.path.dirname(file_path), f"{name_file}_ready.csv")

    # read raw datasets and split label scores and sentences and ignore quotation marks
    df = pd.read_csv(input_path, sep="\t", names=["Sentences", "Label"], quoting=3)

    # truncate sentences to n characters and drop not completely sentences in limit characters
    if char is not None:
        df["Sentences"] = df["Sentences"].str.slice(stop = int(char))

    rows_to_drop = []
    for i in range(len(df)):
        sentence = df['Sentences'].iloc[i]
        # marked index sentences which is not completely sentences
        if sentence[-1].isalpha():
            rows_to_drop.append(i)
    df = df.drop(rows_to_drop).reset_index(drop=True)

    # lowercase for all text in sentences, compose space, only keep words
    df["Sentences"] = df["Sentences"].str.lower()
    df["Sentences"] = df["Sentences"].str.replace(r"[^a-zA-Z]", " ", regex=True)
    df["Sentences"] = df["Sentences"].str.replace(r"\s+", " ", regex=True).str.strip()

    df[["Sentences", "Label"]].to_csv(output_path, index=False)
    return df



def split_data(DataFrame, test_size = 0.2, seed = 123):
    """
    stage 2: split data into train and test set and get ready for ML problem

    :param DataFrame: dataframe ready to split and the size of dataframe should be n x 2[sentence, label]
    :param test_size: chose a split size for dataframe and default has set 30/70 test and train
    :param seed: default is 123, it's random seed make function result unchanged while rerun
    :return:
    """
    x = DataFrame["Sentences"].values
    y = DataFrame["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size,
        random_state=seed, stratify=y)

    train_dataset = [X_train, y_train]
    test_dataset = [X_test, y_test]
    print(f"number of train datasets are {len(X_train)}"
          f"(positive ={sum(y_train == 1)}, negative={sum(y_train == 0)})")
    print(f"number of test datasets are {len(X_test)}"
          f"(positive={sum(y_test == 1)}, negative={sum(y_test == 0)})")
    return train_dataset, test_dataset
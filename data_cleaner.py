import pandas as pd
import os, re, csv

from sklearn.model_selection import train_test_split


def clean_text (dataframe, char = None, rows_drop = False):
    """
    stage 1: clean raw data

    :param file_path: the pathway to find folder which store raw data and reside filtered dataset
    :param name_file: name of the data file
    :param char: keep number of characters in each sentence or row

    :return: write the filtered data file and store at the parent folder as CSV file
    """
    if char is not None:
        dataframe["Sentences"] = dataframe["Sentences"].str.slice(stop = int(char))

    if rows_drop is True:
        rows_to_drop = []
        for i in range(len(dataframe)):
            sentence = dataframe['Sentences'].iloc[i]
            # marked index sentences which is not completely sentences
            if sentence[-1].isalpha():
                rows_to_drop.append(i)
        dataframe = dataframe.drop(rows_to_drop).reset_index(drop=True)

    # lowercase for all text in sentences, compose space, only keep words
    dataframe["Sentences"] = dataframe["Sentences"].str.lower()
    dataframe["Sentences"] = dataframe["Sentences"].str.replace(r"[^a-zA-Z]", " ", regex=True)
    dataframe["Sentences"] = dataframe["Sentences"].str.replace(r"\s+", " ", regex=True).str.strip()
    return dataframe


def text_combined(file_path, file_names, char=None):
    """
    combine multiple datasets

    :param file_path: folder path containing all .txt files
    :param file_names: list of file names (without .txt extension)
                       e.g. ["amazon_cells_labelled", "imdb_labelled", "yelp_labelled"]
    :param char: truncate to N characters (None = no truncation)
    :return: combined cleaned DataFrame
    """
    all_dfs = []

    for name in file_names:
        input_path = os.path.join(file_path, name + ".txt")
        if not os.path.exists(input_path):
            print(f"  [WARNING] file not found: {input_path}, skipped")
            continue

        df = pd.read_csv(input_path, sep="\t", names=["Sentences", "Label"], quoting=3)
        df = clean_text(df, char = char)
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No data files found in {file_path}")

    combined = pd.concat(all_dfs, ignore_index=True)

    # save combined csv
    output_path = os.path.join(os.path.dirname(file_path), "combined_ready.csv")
    combined[["Sentences", "Label"]].to_csv(output_path, index=False)

    print(f"\n  [Combined] total: {len(combined)} reviews "
          f"(positive={sum(combined['Label'] == 1)}, negative={sum(combined['Label'] == 0)})")

    return combined


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
    return train_dataset, test_dataset
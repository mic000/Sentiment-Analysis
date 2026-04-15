import pandas as pd
import os
import nltk

# store filtered datasets into folders
output_folder = "FilteredDatasets"
os.makedirs(output_folder, exist_ok=True)   # make folder if it's not exist

# set raw data input and filtered data output pathway
input_path = os.path.join("RawDatasets", "amazon_cells_labelled.txt")
"""需要gpt看看怎么更好看处理名字"""
output_path = os.path.join(output_folder, "amazon_cells_filtered.csv")

# read raw datasets and split label scores and sentences
df = pd.read_csv(input_path, sep="\t", names=["Sentences", "Label"])

# truncate sentences to 50 characters and drop not completely sentences in limit characters
df['Sentences'] = df['Sentences'].str.slice(stop=50)
rows_to_drop = []
for i in range(len(df)):
    sentence = df['Sentences'][i]
    # marked index sentences which is not completely sentences
    if sentence[-1] != ".":
        rows_to_drop.append(i)
df = df.drop(rows_to_drop)

# lowercase for all text in sentences
df['Sentences'] = df['Sentences'].str.lower()




# write the filtered dataframe to a new CSV file
# df[['Sentences', 'Label']].to_csv(output_path, index=False)
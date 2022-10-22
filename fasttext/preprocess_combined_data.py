import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict, Split
import textcleaner


db_dir = "/home/daniel/data/uni/masterarbeit-sentiment/data/datasets/experiments/de/3sentiment-exact"
db_file = f"{db_dir}/datasets.db"

# remove data samples that are duplicates and not in the same split
remove_duplicates = False

# Get combined dataset
con = sqlite3.connect(db_file)

if not remove_duplicates:
    df_combined = pd.read_sql("SELECT * FROM dataset", con=con)
else:
    # remove own data "oliverguhr/*"
    # also remove all duplicates from "umsab/german" that are not in the same split
    df_combined = pd.read_sql('SELECT * FROM dataset WHERE NOT dataset LIKE "oliverguhr/%" AND (NOT dataset == "umsab/german" OR (dataset == "umsab/german" AND id NOT IN (SELECT `umsab/german` as id FROM duplicates WHERE same_split=0)))', con=con)

con.close()

# Preprocess and export combined dataset
dataset_combined = DatasetDict()

dataset_combined["train"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "train"],
    split=Split.TRAIN
)
dataset_combined["test"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "test"],
    split=Split.TEST
)
dataset_combined["validation"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "dev"],
    split=Split.VALIDATION
)


def preprocess_combined(sample):
    # text preprocessing
    sample["text"] = textcleaner.cleanText(sample["text"])

    # convert to ttlab labels
    ttlab_mapping = {
        1: 0,
        -1: 1,
        0: 2
    }
    sample["labels"] = ttlab_mapping[sample["ttlab_label"]]

    return sample


print(dataset_combined["train"][0])
print(dataset_combined["train"][10])
print(dataset_combined["train"][565])
print(dataset_combined["train"][110])
print(dataset_combined["train"][1110])
print(dataset_combined["train"][56115])

dataset_combined = dataset_combined.map(preprocess_combined)

dataset_combined = dataset_combined.remove_columns(['original_label', 'ttlab_label', 'sentiment', 'split', '__index_level_0__'])

print(dataset_combined["train"][0])
print(dataset_combined["train"][10])
print(dataset_combined["train"][565])
print(dataset_combined["train"][110])
print(dataset_combined["train"][1110])
print(dataset_combined["train"][56115])

combined_dataset_name = "oliverguhr"
if remove_duplicates:
    combined_dataset_name += "-unseen"
combined_dataset_path = f"/home/daniel/data/uni/masterarbeit-sentiment/data/datasets/experiments/de/3sentiment-exact/{combined_dataset_name}"

# save preprocessed dataset
dataset_combined.save_to_disk(combined_dataset_path)

import csv
import sqlite3
import pandas as pd


label_list = ["positive", "negative", "neutral"]
label_map = {label : i for i, label in enumerate(label_list)}

ttlab_mapping = [+1, -1, 0]


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if len(line) > 0:
                lines.append(line)
        return lines


def _create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        #if len(line[1]) > 128:
        #    continue #skip lines that are too long. otherwhise they will be truncated
        # count tokens, not chars!!!
        guid = "%s-%s" % (set_type, i)

        # use raw unprocessed text
        #text_a = line[1].replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")
        text_a = line[2].strip()

        if text_a:
            # dataset
            dataset_name = line[0].strip()

            # create labels
            label = line[1].replace("__label__","")

            original_label = label_map[label]
            ttlab_label = ttlab_mapping[original_label]

            examples.append({
                "text": text_a,
                "original_label": original_label,
                "ttlab_label": ttlab_label,
                "split": set_type,
                "id": guid,
                "sentiment": label,
                "dataset": f"oliverguhr/{dataset_name}"
            })

    examples = pd.DataFrame(examples)

    print("created {} examples".format(len(examples)))
    return examples


if __name__ == "__main__":
    df_train = _create_examples(
        _read_tsv("/home/daniel/data/uni/masterarbeit-sentiment/extern/german-sentiment/fasttext/modeldata/model.train"),
        "train"
    )

    df_dev = _create_examples(
        _read_tsv("/home/daniel/data/uni/masterarbeit-sentiment/extern/german-sentiment/fasttext/modeldata/model.valid"),
        "dev"
    )

    df_test = _create_examples(
        _read_tsv("/home/daniel/data/uni/masterarbeit-sentiment/extern/german-sentiment/fasttext/modeldata/model.test"),
        "test"
    )

    df = pd.concat([df_train, df_dev, df_test])
    df.set_index("id", inplace=True)

    db_dir = "/home/daniel/data/uni/masterarbeit-sentiment/data/datasets/experiments/de/3sentiment-exact"
    db_file = f"{db_dir}/datasets.db"

    con = sqlite3.connect(db_file)
    df.to_sql("dataset", con=con, index=True, index_label="id", if_exists='append')
    con.close()

    print("done")

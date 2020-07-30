import pickle
import pandas as pd

if __name__ == "__main__":
    data_path = "/Users/dheerajmekala/Work/DiscPattern/data/agnews/"
    label_dic = {1: "politics", 2: "sports", 3: "business", 4: "technology"}
    df = pd.read_csv(data_path + "dataset.csv", header=None)

    texts = []
    labels = []
    for i, row in df.iterrows():
        label = label_dic[row[0]]
        text = row[1] + " . " + row[2]
        labels.append(label)
        texts.append(text.lower())

    new_df = pd.DataFrame.from_dict({"text": texts, "label": labels})
    pickle.dump(new_df, open(data_path + "df.pkl", "wb"))
    pass

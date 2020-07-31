import pickle
import json
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from word2vec import fit_get_tokenizer
import numpy as np


def make_index(y_train, label_to_index):
    y_inds = []
    for i in y_train:
        y_inds.append(label_to_index[i])
    return y_inds


def make_name(y_pred_index, index_to_label):
    y_names = []
    for i in y_pred_index:
        y_names.append(index_to_label[i])
    return y_names


def build_vocab(tokenizer):
    word_index = {}
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
        word_index[w] = tokenizer.word_index[w]
    return word_index, index_word


def BOW(texts, tokenizer, index_word):
    vocab_size = len(index_word)
    input_arr = np.zeros((len(texts), vocab_size + 1))
    for i, text in enumerate(texts):
        tokens = tokenizer.texts_to_sequences([text])[0]
        for tok in tokens:
            input_arr[i][tok] = 1
    return input_arr


def train(df, tokenizer):
    word_index, index_word = build_vocab(tokenizer)
    labels = set(df.label)
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.1, random_state=42)
    bow_train = BOW(X_train, tokenizer, index_word)
    bow_test = BOW(X_test, tokenizer, index_word)
    y_train_index = make_index(y_train, label_to_index)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(bow_train, y_train_index)
    y_pred_index = clf.predict(bow_test)
    y_pred = make_name(y_pred_index, index_to_label)
    print(classification_report(y_test, y_pred))
    return clf


if __name__ == "__main__":
    base_path = "/data4/dheeraj/discpattern/"
    # base_path = "/Users/dheerajmekala/Work/DiscPattern/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    df = pickle.load(open(data_path + "df.pkl", "rb"))
    with open(data_path + "seedwords.json") as fp:
        label_term_dict = json.load(fp)
    try:
        tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    except:
        tokenizer = fit_get_tokenizer(df.text, max_words=150000)

    clf = train(df, tokenizer)
    pickle.dump(clf, open(base_path + "clf_dt.pkl", "wb"))
    tree.plot_tree(clf)

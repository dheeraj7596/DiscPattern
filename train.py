import pickle
import os
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from util import *
import json
import pandas as pd


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        for l in count_dict:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


def train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, dataset_path):
    print("Going to train classifier..")
    basepath = dataset_path
    model_name = "discpattern"
    dump_dir = basepath + "models/" + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))

    X, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)

    dic = {"Text": X, "Pseudo Label": y, "True Label": y_true}
    pseudo_df = pd.DataFrame.from_dict(dic)
    pickle.dump(pseudo_df, open(data_path + "pseudo_df.pkl", "wb"))

    y_one_hot = make_one_hot(y, label_to_index)
    print("Fitting tokenizer...")
    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                      max_sentences=max_sentences,
                                                      max_sentence_length=max_sentence_length,
                                                      max_words=max_words)
    print("Creating Embedding matrix...")
    embedding_matrix = pickle.load(open(dataset_path + "embedding_matrix.pkl", "rb"))
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
    print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
    X_all = prep_data(texts=df["text"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    y_true_all = df["label"]
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true_all, pred_labels))
    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels


if __name__ == "__main__":
    # create pseudo labels from the seed words
    # train the classifier
    # test it on data
    # analyze the wrong predictions

    base_path = "/Users/dheerajmekala/Work/DiscPattern/data/"
    dataset = "agnews"
    data_path = base_path + dataset + "/"

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    with open(data_path + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    labels = list(label_term_dict.keys())
    label_to_index = {}
    index_to_label = {}

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label

    pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label, data_path)

    texts = []
    mis_pred_labels = []
    true_labels = []

    for i, row in df.iterrows():
        if row["label"] == pred_labels[i]:
            continue
        texts.append(row["text"])
        mis_pred_labels.append(pred_labels[i])
        true_labels.append(row["label"])

    dic = {"Text": texts, "Prediction": mis_pred_labels, "True Label": true_labels}
    mis_pred_df = pd.DataFrame.from_dict(dic)
    pickle.dump(mis_pred_df, open(data_path + "mis_pred_df.pkl", "wb"))

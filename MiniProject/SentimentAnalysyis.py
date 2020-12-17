import re
import os
import matplotlib.pyplot as plt
import keras
import wget
import pydot
import nltk
import numpy as np
import zipfile
import pandas as pd
from os import path
import seaborn as sns
import tensorflow as tf
import pickle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
nltk.download('brown')
nltk.download('reuters')
from nltk.corpus import stopwords, words, brown, reuters
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense, LSTM, Dropout, GRU, Bidirectional
from keras import regularizers
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import sys

np.set_printoptions(threshold=sys.maxsize)
pd.options.mode.chained_assignment = None


# Clean excel file of unnecessary rows and columns
def clean_excel(data: pd.DataFrame):
    # Preprocessing Training data
    # Removing first row as it has only information about class value mappings
    data.rename(columns={'Unnamed: 4': 'target'}, inplace=True)
    # Dropping empty columns
    data.dropna(how='all', inplace=True, axis=1)
    data.dropna(subset=['Anootated tweet', 'target'], inplace=True, axis=0)
    return data


# Only include classes specified in accepted_class
def clean_class(df, accepted_class: np.ndarray):
    # accepted_class = np.array([1, 0, -1])
    decode_map = {0: 0, 2: 2, 1: 1, -1: -1, '0': 0, '2': 2, '1': 1, '-1': -1}
    df['target'].map(decode_map)
    df = df.loc[df['target'].isin(accepted_class)]
    df.astype({'target': 'int'}).dtypes
    return df


# Preprocessing text
def preprocess_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    # removing Http Links
    text = re.sub(r'https?:\S+', '', text)
    # removing special characters
    text = re.sub(r'[^\w\s]', '', text)
    # text = re.sub(r'\b[0-9]+\b','',text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    text = ' '.join(split_sentence_of_text(text))
    text = ' '.join(wordnet_lemmatizer.lemmatize(word, 'v') for word in text.split())
    return text.strip()


# Compound Splitting of words
def split_hashtag_to_words_all_possibilities(hashtag):
    all_possibilities = []

    split_possibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_possibility) if x == True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)

            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]

    return all_possibilities


def split_sentence_of_text(sentence):
    digit_regex = re.compile(r"\d")
    sentence_split = sentence.split(' ')
    result = []

    for word in sentence_split:
        word_to_split = word
        if word_to_split not in word_dictionary and word_to_split.rstrip('s') not in word_dictionary:

            num_if_any = ""
            if digit_regex.search(word):
                char_num = re.match(r'([a-zA-Z]+)([0-9]+)', word)
                if char_num:
                    word_to_split = char_num.group(1)
                    num_if_any = char_num.group(2)
                else:
                    num_char = re.match(r'([0-9]+)([a-zA-Z]+)', word)
                    if num_char:
                        word_to_split = num_char.group(2)
                        num_if_any = num_char.group(1)
            try:
                if not re.match(r'[0-9]+$', word_to_split):
                    split = split_hashtag_to_words_all_possibilities(word_to_split)
                    if (split):
                        result += split[0]
                else:
                    result.append(word_to_split)
            except:
                result.append(word_to_split)
            if num_if_any:
                result.append(num_if_any)
        else:
            result.append(word_to_split)

    return result


def load_glove_embeddings():
    if not path.exists('glove.twitter.27B.zip'):
        print('Please wait while the Global Vector Word Embeddings are being downloaded.')
        wget.download('http://nlp.stanford.edu/data/glove.twitter.27B.zip')
    glove = zipfile.ZipFile('./glove.twitter.27B.zip')
    embeddings = {}
    # with open('../input/glovetwitter/glove.twitter.27B.200d.txt', 'r') as file:
    with glove.open('glove.twitter.27B.200d.txt', 'r') as file:
        for line in file:
            line = line.decode('utf-8')
            line_tokens = line.split()
            word = line_tokens[0]
            word_vectors = np.asarray(line_tokens[1:], dtype='float32')
            embeddings[word] = word_vectors
    return embeddings


def generate_embeddings(tokenizer, word_embedding):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_size = 200
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, index in tokenizer.word_index.items():
        vector = word_embedding.get(word)
        if vector is not None:
            embedding_matrix[index] = vector
    return embedding_matrix


def prediction_labels(pred_labels):
    l = np.argmax(pred_labels, axis=1)
    l = np.where(l == 0, -1, l)
    l = np.where(l == 1, 0, l)
    l = np.where(l == 2, 1, l)
    return l


# Function to plot the Accuracy and Loss of the training and testing models

def generate_plot(model, path):
    plt.plot(model.history['categorical_accuracy'])
    plt.plot(model.history['val_categorical_accuracy'])
    plt.title('Model Accuracy (Train & Validation)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(f'{path}/val_categorical_accuracy.png')
    plt.show()

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss(Train & Validation)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{path}/val_loss.png')
    plt.show()


def split_train_test(data, flag):
    if (flag):
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['Anootated tweet'], data['target'],
                                                            test_size=0.2, stratify=data['target'])
        return X_train, X_test, y_train, y_test
    else:
        return 0


def tokenize(data):
    tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
    tokenizer.fit_on_texts(data)
    return tokenizer


# https://scikit-learn.org/0.21/auto_examples/model_selection/
# plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes, path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f'{path}/confusion_matrix.png')
    plt.show()


def evaluate_test_data(model, y_test, x_test_processed, model_path):
    print("Test data evaluation")
    model.evaluate(x_test_processed, pd.get_dummies(y_test))
    test_pred_labels = model.predict(x_test_processed)
    test_pred_labels = prediction_labels(test_pred_labels)
    classes = unique_labels(test_pred_labels, y_test.astype(int))
    print("Test data Evaluation")
    cf = confusion_matrix(y_test.astype(int), test_pred_labels, classes)
    print(classification_report(y_test.astype(int), test_pred_labels))
    plot_confusion_matrix(cf, classes, model_path)
    return test_pred_labels


def train_evaluate_model(stack_length=1, **kwargs):
    data = kwargs['data']
    model_type = kwargs['model_type']
    if model_type not in [LSTM, SimpleRNN, GRU, Bidirectional]:
        print(f"Invalid model {model_type} defaulting to LSTM")
        model_type = LSTM
    input_len = kwargs['input_length']
    word_embeddings = kwargs['word_embeddings']
    model_units = kwargs['model_units']
    epochs = kwargs['epochs']
    learning_rate = kwargs['learning_rate']
    train_on_all_data = kwargs['train_on_all_data']
    early_stopping = kwargs['early_stopping']
    stacked_model = kwargs['stacked_model']
    # stack_length = kwargs['stack_length']
    # reduce_lr_on_plateau = kwargs['reduce_lr_on_plateau']

    base_dir = str(model_type).strip('<>\'').split('.')[-1]
    base_path = f'./{base_dir}/units{model_units}/stacks_{stack_length}/Estop{early_stopping}'
    
    metric = 'val_loss'    
    
    checkpoint_filepath = f'{base_path}/checkpoint'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        # save_weights_only=True,
        save_freq='epoch',
        verbose=1,
        monitor=metric,
        mode='min',
        save_best_only=True)
    

    # Callbacks default model_checkpoint ,early_stopping and reduce_lr_on_plateau must be passed as kwargs to
    # train_evaluate_model
    my_callbacks = [model_checkpoint]
    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=metric, mode='min',
                                                                   restore_best_weights=True, patience=5, verbose=1)
        my_callbacks.append(early_stopping_callback)
    print(f"**{base_dir} early_stopping={early_stopping},stacked_model={stacked_model},stack_length={stack_length}****")
    if not train_on_all_data:
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = split_train_test(data, not train_on_all_data)
        tokenizer = tokenize(X_train)
        word_embedding_matrix = generate_embeddings(tokenizer, word_embeddings)
        X_train_processed = tokenizer.texts_to_sequences(X_train)
        X_train_processed = pad_sequences(X_train_processed, padding='post', maxlen=50, truncating='pre')
        X_test_processed = tokenizer.texts_to_sequences(X_test)
        X_test_processed = pad_sequences(X_test_processed, padding='post', maxlen=50, truncating='pre')
        
    else:
        tokenizer = tokenize(data['Anootated tweet'])
        pickle.dump(tokenizer,open("./vectorizer","wb"))
        word_embedding_matrix = generate_embeddings(tokenizer, word_embeddings)
        X_train_processed = tokenizer.texts_to_sequences(data['Anootated tweet'])
        X_train_processed = pad_sequences(X_train_processed, padding='post', maxlen=50, truncating='pre')
        y_train = data['target']
        

    model = Sequential()

    embedding_layer = Embedding(len(tokenizer.word_index) + 1, 200, weights=[word_embedding_matrix],
                                input_length=input_len,
                                trainable=False, mask_zero=True)
    model.add(embedding_layer)

    if stack_length <= 0:
        print(f"Stack length cannot be {stack_length} defaulting to 1")
        stack_length = 1
    if model_type != Bidirectional:
        if stack_length == 1:
            model.add(model_type(model_units, kernel_regularizer=regularizers.l2(l2=0.01)))
        else:
            model.add(model_type(model_units, kernel_regularizer=regularizers.l2(l2=0.01), return_sequences=True))
            i = 0
            for i in range(stack_length - 2):
                model.add(model_type(model_units, kernel_regularizer=regularizers.l2(l2=0.01), return_sequences=True))
            model.add(model_type(model_units, kernel_regularizer=regularizers.l2(l2=0.01)))
    elif model_type == Bidirectional:
        if stack_length == 1:
            model.add(model_type(LSTM(units=model_units, kernel_regularizer=regularizers.l2(l2=0.01))))
        else:
            model.add(
                model_type(LSTM(units=model_units, kernel_regularizer=regularizers.l2(l2=0.01), return_sequences=True)))
            for i in range(stack_length - 2):
                model.add(model_type(
                    LSTM(units=model_units, kernel_regularizer=regularizers.l2(l2=0.01), return_sequences=True)))
            model.add(model_type(LSTM(units=model_units, kernel_regularizer=regularizers.l2(l2=0.01))))

    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())

    dot_img_file = f'{base_path}/model.png'

    model_history = model.fit(X_train_processed, pd.get_dummies(y_train), batch_size=128, epochs=epochs, verbose=1,
                              validation_split=0.2, callbacks=my_callbacks)
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    if not train_on_all_data:
        generate_plot(model_history, base_path)
        test_pred_labels = evaluate_test_data(model, y_test, X_test_processed, base_path)
        print(
            "**************************************************************************************************************************************************")
        return model, test_pred_labels
    else:
        print(
            "**************************************************************************************************************************************************")
        return model


def write_file(output, file_name, open_mode='w'):
    with open(file_name, open_mode) as file:
        file.write(output)
    file.close()


if __name__ == '__main__':

    # Creating word dictionary from multiple word corpus from nltk

    word_dictionary = set(brown.words())
    word_dictionary = word_dictionary.union(set(reuters.words()))
    word_dictionary = word_dictionary.union(set((list(wn.words()))))
    word_dictionary = word_dictionary.union(
        {"obama", "romney", "ryan", "barack", "mitt", "michelle", "anne", "joe", "biden", "paul"})

    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        if alphabet in word_dictionary:
            word_dictionary.remove(alphabet)


#     print("Loading Glove Embeddings") 
#     glove_embedding = load_glove_embeddings()
    
    
#     # Read data from all the sheets
#     sheets = [0, 1]
#     # input_data = pd.read_excel('../input/cs583-dataset/training-Obama-Romney-tweets.xlsx', sheet_name=sheets)
#     print("Reading data")
#     input_data = pd.read_excel('training-Obama-Romney-tweets.xlsx', sheet_name=sheets)
#     # Clean excel remove unecessary rows and emtpy columns
#     data_clean = pd.DataFrame()
#     for i in sheets:
#         data_temp = data_temp[1:]
#         data_temp = clean_excel(input_data[i])
#         data_clean = data_clean.append(data_temp)
#     data_clean = clean_class(data_clean, np.array([1, 0, -1]))

#     # Text Preprocessing
#     print("Preprocessing")
#     data_clean['Anootated tweet'] = data_clean['Anootated tweet'].apply(lambda x: preprocess_text(x))
#     target_count = data_clean['target'].value_counts().reset_index()
    

#     ax = sns.barplot(x=target_count['index'], y=target_count['target'])
#     plt.title("Tweet Distribution")
#     plt.show()

#     # model_type allowed
#     # LSTM, SimpleRNN, GRU, Bidirectional, case sensitive

#     # Best model selected
#     # model_GRU_2_2, GRU_pred_labels_2_2 = train_evaluate_model(data=data_clean,
#     #                                                     model_type=GRU,
#     #                                                     word_embeddings=glove_embedding,
#     #                                                     input_length=50,
#     #                                                     epochs=50,
#     #                                                     learning_rate=0.001,
#     #                                                     model_units=100,
#     #                                                     train_on_all_data=False,
#     #                                                     early_stopping=True,
#     #                                                     stacked_model=False
#     #                                                     )



# #Re train the best model on whole dataset 
#     best_model = train_evaluate_model(data=data_clean,
#                                                         model_type=GRU,
#                                                         word_embeddings=glove_embedding,
#                                                         input_length=50,
#                                                         epochs=50,
#                                                         learning_rate=0.001,
#                                                         model_units=100,
#                                                         train_on_all_data=True,
#                                                         early_stopping=True,
#                                                         stacked_model=False
#                                                         ) 

    # best_model.save('best_model')

# read test data 
    model =keras.models.load_model("best_model")
    tokenizer = pickle.load(open('vectorizer','rb'))


    obama_data = pd.read_excel('final-testData-no-label-Obama-tweets(1).xlsx',header=None)
    romney_data = pd.read_excel('final-testData-no-label-Romney-tweets(1).xlsx',header=None)
    obama_data[1] = obama_data[1].apply(lambda x: preprocess_text(x))
    romney_data[1] = romney_data[1].apply(lambda x: preprocess_text(x))

    #Tokenize test dataset
    obama_tweets = tokenizer.texts_to_sequences(obama_data[1])
    romney_tweets = tokenizer.texts_to_sequences(romney_data[1])
    #Pad test data sequences
    obama_tweets=pad_sequences(obama_tweets, padding='post', maxlen=50, truncating='pre')
    romney_tweets=pad_sequences(romney_tweets, padding='post', maxlen=50, truncating='pre')

    obama_data['predicted_class'] = prediction_labels(model.predict(obama_tweets))
    romney_data['predicted_class']= prediction_labels(model.predict(romney_tweets))
    
    
    
    student_id ='57'
    write_file(f"{student_id} \n", './obama.txt','w')
    for i in range(len(obama_data)):
        if(i != len(obama_data)-1):
            write_file(f"{obama_data.iloc[i][0]};;{obama_data.iloc[i]['predicted_class']}\n",'./obama.txt','a')
        else:
            write_file(f"{obama_data.iloc[i][0]};;{obama_data.iloc[i]['predicted_class']}",'./obama.txt','a')


    write_file(f"{student_id} \n", './romney.txt','w')
    for i in range(len(romney_data)):
        if(i != len(romney_data)-1):
            write_file(f"{romney_data.iloc[i][0]};;{romney_data.iloc[i]['predicted_class']}\n",'./romney.txt','a')
        else:
            write_file(f"{romney_data.iloc[i][0]};;{romney_data.iloc[i]['predicted_class']}",'./romney.txt','a')

   
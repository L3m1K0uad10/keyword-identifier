import os

import pandas as pd
import tensorflow as tf 
import spacy



# Determine the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file
csv_path = os.path.join(script_dir, "..", "data", "keywords.csv")


# Load the spaCy English model for additional text cleaning (if needed)
# Make sure you have run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


df = pd.read_csv(csv_path)

# split dataset
train_df = df.head(173) # training dataset
eval_df = df.tail(24) # evaluation dataset

# print(eval_df.duplicated(keep = False))

# tokens 
train_tokens = train_df["token"].astype(str).values # Ensuring tokens are string before retrieving the values
eval_tokens = eval_df["token"].astype(str).values

# labels
train_label = train_df.pop(item = "is_keyword")
eval_label = eval_df.pop(item = "is_keyword")

# print(eval_df_label)

# text vectorization
# help in preprocessing layer which maps text features to integer sequences
vectorizer = tf.keras.layers.TextVectorization(
    standardize = "lower_and_strip_punctuation",
    split = "character",
    ngrams = 2,
    output_mode = "tf_idf",
    max_tokens = 5000
)

# adapt the vectorizer to the token data
# NOTE: we do not adapt() the eval tokens, only the train tokens
vectorizer.adapt(train_tokens)

X = vectorizer(train_tokens)
X_ = vectorizer(eval_tokens)
#print("Feature matrix shape: ", X_.shape)
# print("a sample vectorized representation for the frist token: ", X_[0])
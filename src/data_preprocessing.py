import pandas as pd



df = pd.read_csv("../data/keywords.csv")

# split dataset
train_df = df.head(173) # training dataset
eval_df = df.tail(24) # evaluation dataset

# print(eval_df.duplicated(keep = False))

# labels
train_df_label = train_df.pop(item = "is_keyword")
eval_df_label = eval_df.pop(item = "is_keyword")

# print(eval_df_label)

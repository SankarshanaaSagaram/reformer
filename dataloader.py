import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ReformerForSequenceClassification

model = ReformerForSequenceClassification.from_pretrained('google/reformer-crime-and-punishment')
num_labels = 3

model.classifier = nn.Linear(model.config.hidden_size, num_labels)

tokenizer = AutoTokenizer.from_pretrained('google/reformer-crime-and-punishment')
tokenizer.pad_tokem = '<pad>'

selected_text_list_train = train['selected_text'].astype(str).tolist()

train_encodings = tokenizer.batch_encode_plus(
    selected_text_list_train,
    max_length = 2**19,
    padding = 'max_length',
    truncation = True,
    #return_attention_mask = True
    #return_tensor = 'pt'
)

train_labels = torch.tensor(train['sentiment'].astype('category').cat.codes.tolist())

selected_text_list_test = test['selected_text'].astype(str).tolist()

test_encodings = tokenizer.batch_encode_plus(
    selected_text_list_test,
    max_length=2 ** 19 ,
    padding='max_length',
    truncation=True,
    #return_attention_mask=True,
    #return_tensors='pt'
)

test_labels = torch.tensor(test['sentiment'].astype('category').cat.codes.tolist())

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    train_labels
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    test_labels
)

# count the number of instances for each class
class_counts = train["sentiment"].value_counts()

# calculate the weight of each class
class_weights = {
    "positive": class_counts["positive"] / sum(class_counts),
    "negative": class_counts["negative"] / sum(class_counts),
    "neutral": class_counts["neutral"] / sum(class_counts)
}

# convert the class weights dictionary into a tensor
class_weights_tensor = torch.tensor([
    class_weights["positive"],
    class_weights["negative"],
    class_weights["neutral"]
])

# push to GPU
#weights = weights.to(device)
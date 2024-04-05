import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, TFBertModel
from sklearn.metrics import accuracy_score
import pandas as pd
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

#Loading Events Data
with open('/content/Events_Data_Dataset.json', 'r') as json_file:
    events_data = json.load(json_file)

#Initialising an empty list to store events data
events_data_list = []

#Iterating through each event in the JSON data
for event in events_data:

    event_data = {
        "event_id": event["event_id"],
        "event_parameters": " ".join(param["parameter_name"] for param in event["event_parameters"])
    }

    events_data_list.append(event_data)

#Creating DataFrame from the list
events_df = pd.DataFrame(events_data_list)
print(events_df)

#Preprocessing
events_df['cleaned_params'] = events_df['event_parameters'].apply(lambda x: x.lower())
events_df['cleaned_params'] = events_df['cleaned_params'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

#Implementing Tokenizer and BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

#Cosine Similarity function
def get_bert_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    return tf.reduce_mean(outputs['last_hidden_state'], axis=1).numpy()

def get_cosine_similarity(input_text, corpus, tokenizer, bert_model):
    input_embedding = get_bert_embedding(input_text, tokenizer, bert_model)
    corpus_embeddings = np.array([get_bert_embedding(text, tokenizer, bert_model) for text in corpus])

    input_embedding = input_embedding.reshape(1, -1)
    corpus_embeddings = corpus_embeddings.reshape(len(corpus), -1)

    similarity_scores = cosine_similarity(input_embedding, corpus_embeddings).flatten()

    result_df = pd.DataFrame({'Similarity': similarity_scores, 'Original': events_df['event_parameters']})
    result_df = result_df.sort_values(by='Similarity', ascending=False)

    return result_df

#Splitting data into train and test sets
print(train_data)
print(test_labels)

#Defining DataLoader
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(input_data, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(input_data), torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

#Defining BERT model for sequence classification
from transformers import BertForSequenceClassification

def get_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return model

#Training loop
def train_model(model, train_dataloader, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

#Hyperparameter Tuning
print(train_data.index)

class BertClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, lr=5e-5, batch_size=32, num_labels=2, num_epochs=10):
        self.lr = lr
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            # print("Converted")
            X = pd.DataFrame({'text': X})
            # print(X)
            # print(X.index)

        train_dataloader = self.get_dataloader(X['text'].tolist(), y, shuffle=True)

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for inputs, labels in train_dataloader:
                optimizer.zero_grad()

                inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

                if labels is not None:
                    labels = torch.tensor(labels, dtype=torch.long)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {average_loss}")

        return self

    def transform(self, X):
        return X

    def predict(self, X):
        test_dataloader = self.get_dataloader(X, labels=None, shuffle=False)

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for inputs, _ in test_dataloader:
                inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())

        return all_predictions

    def get_dataloader(self, input_data, labels, shuffle=True):
        input_tensors = [self.tokenizer(text, return_tensors='pt', padding=True, truncation=True) for text in input_data]
        input_ids = torch.stack([tensor['input_ids'].squeeze(0) for tensor in input_tensors])
        attention_masks = torch.stack([tensor['attention_mask'].squeeze(0) for tensor in input_tensors])

        print(input_ids)
        print(labels)
        dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

param_grid = {
    'lr': [5e-5, 3e-5, 2e-5],
    'batch_size': [16, 32, 64],
}

bert_classifier = BertClassifier(num_epochs=10)

grid_search = GridSearchCV(bert_classifier, param_grid, cv=3, scoring='accuracy', verbose=1,error_score='raise')
grid_search.fit(train_data, train_labels)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

final_model = grid_search.best_estimator_

#Model Optimisation and Fine-Tuning
train_dataloader = get_dataloader(train_data, train_labels, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_data, val_labels, batch_size=32, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

model.eval()
val_accuracy = evaluate_model(model, val_dataloader)
print(f"Validation Accuracy: {val_accuracy}")

#Accuracy calculation after optimisation
def evaluate_model(model, test_dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

#Ensemble Learning
def train_model_with_seed(seed):
    torch.manual_seed(seed)
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model

num_models = 5
models = [train_model_with_seed(seed) for seed in range(num_models)]

def combine_predictions(models, input_data):
    all_predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(input_data, dtype=torch.long)
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())

    combined_predictions = np.mean(all_predictions, axis=0)
    return combined_predictions

combined_predictions = combine_predictions(models, test_data)

combined_accuracy = accuracy_score(test_labels, combined_predictions)
print(f"Ensemble Accuracy: {combined_accuracy}")

#Gradient Accumulation
accumulation_steps = 2

model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0

    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs.loss
        loss.backward()

        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

#END OF CODE

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# # %%
import numpy as np
import pandas as pd
import torch
import re
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Module, GRU, Embedding, Linear, Sigmoid, CrossEntropyLoss
# # Part 1

# from google.colab import drive
# drive.mount('/content/drive')


"data/sentiment_analysis/train_pos_merged.txt"
# function clearing HTML tags from text
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# preprocessing
def clean_text(path):
    reviews = []
    all_words = []
    with open(path) as pos:
        lines = pos.readlines()
        for line in lines:
            #clear html tags
            line = cleanhtml(line)
            # lower case and punctuation
            line = re.sub(r'[^a-zA-Z]', ' ', line.lower())
            # split to list of words
            words = line.split()
            # add list to reviews
            reviews.append(words)
            # extend words with new review
            all_words.extend(words)

    return reviews, all_words

def create_vocab(words):
    # create vocabulary with indexes
    vocab = {}
    id = 1
    for word in words:
        if word not in vocab.keys():
            vocab[word] = id
            id += 1
    return vocab


def vectorize_data(reviews, y, vocab, LENGTH=400):
    y = np.array([y for _ in range(len(reviews))])
    indexed_reviews = np.zeros((len(reviews), LENGTH), dtype = np.int64)
    for i, review in enumerate(reviews):
        indexed_review = []
        for word in review:
            indexed_review.append(vocab[word])
        indexed_reviews[i, max(LENGTH-len(review),0):] = indexed_review[:400]
    return indexed_reviews, y

def preprocessing(path1, path2, y1, y2, vocab, LENGTH=400):
    reviews1, words1 = clean_text(path1)
    reviews2, words2 = clean_text(path2)
    # words1.extend(words2)
    # print(words1)

    del words1, words2

    # vocab = create_vocab(words1)

    x1, y1 = vectorize_data(reviews1, y1, vocab, LENGTH)
    x2, y2 = vectorize_data(reviews2, y2, vocab, LENGTH)

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    return x, y #, vocab



        


# !ls /content/drive
# path = "/content/drive/MyDrive/Deep_Learning/sentiment_analysis/"
path = "./data/sentiment_analysis/"
reviews, words = clean_text(path + "all_merged.txt")


vocab = create_vocab(words)
# vocab


train_x, train_y = preprocessing(path + "train_pos_merged.txt", path + "train_neg_merged.txt", 0, 1, vocab)


# input = torch.from_numpy(train_x[0])
# embedding = Embedding(len(vocab), 3, padding_idx=0)
# embedding(input)


batch_size = 100
train_data =  TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data)

# # Part 2

class GRU_model(Module):

    def __init__(self, vocab_size, input_dim, hidden_dim, n_layers=1, LENGTH=400):
        
        super(GRU_model, self).__init__()
        
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = Embedding(vocab_size, input_dim, padding_idx=0)
        self.gru = GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = Linear(hidden_dim, 2)
        self.sigmoid = Sigmoid()

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        # print(f"shape of x: {x.shape}; shape of h: {h.shape}; shape of x[:,-1]: {x[:,-1].shape}")
        x = self.linear(x[:,-1])
        # print(f"shape of x: {x.shape}")
        x = self.sigmoid(x)
        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def gru_train(train_loader, vocab_size, learn_rate, input_dim=10, hidden_dim=16, EPOCHS=5):
    
    # Setting common hyperparameters
    # input_dim = next(iter(train_loader))[0].shape[1]
    # print(next(iter(train_loader))[0].shape[1])
    output_dim = 1
    n_layers = 1
    # Instantiating the model
    model = GRU_model(vocab_size, input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    print("Starting Training")
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()
            
            out, h = model(x.to(device), h)
            # print(f"shape of out.squeeze(): {out.squeeze().shape}; shape of label: {label.shape}")
            loss = criterion(out.squeeze(), label.to(device))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def gru_evaluate(model, test_loader): #, label_scalers):
    model.eval()
    outputs = []
    results = []
    start_time = time.time()
    model.eval()
    err = 0
    for x, label in test_loader:
        h = model.init_hidden(test_loader.batch_size).data
        input = x.to(device)
        output, h_out = model(input, h)
        result = torch.argmax(output, dim=1)
        results.append(result)
        err += torch.abs(result.to(device) - label.to(device)).sum()
    accuracy = 1 - err/len(test_loader)
    return accuracy, outputs, results


gru_model = gru_train(
    train_loader, 
    vocab_size = len(vocab), 
    learn_rate=0.0005, 
    hidden_dim=32, 
    EPOCHS=300
    )


test_x, test_y = preprocessing(path + "test_pos_merged.txt", path + "test_neg_merged.txt", 0, 1, vocab)


batch_size = 100
test_data =  TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
test_loader = DataLoader(test_data) #, shuffle=True, batch_size=batch_size)



gru_model = GRU_model(len(vocab), 10, 32, 1, 1)
gru_model.load_state_dict(torch.load("./models/gru/gru_rnn.pt", map_location=device))


test_accuracy, test_outputs, test_results = gru_evaluate(gru_model, test_loader)
train_accuracy, train_outputs, train_results = gru_evaluate(gru_model, train_loader)


train_accuracy
 
# ## **Part 3** MLP

class MLP_model(Module):

    def __init__(self, vocab_size, input_dim, hidden_dim, LENGTH = 400):
        
        super(MLP_model, self).__init__()

        self.embedding = Embedding(vocab_size, input_dim, padding_idx=0)
        self.fc1 = Linear(input_dim*LENGTH, hidden_dim)
        self.fc2 = Linear(hidden_dim, 2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def mlp_train(train_loader, vocab_size, learn_rate, input_dim=10, hidden_dim=16, EPOCHS=5):
    
    # Instantiating the model
    model = MLP_model(vocab_size, input_dim, hidden_dim)
    # print(model)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    print("Starting Training")
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            model.zero_grad()
            
            out = model(x.to(device))
            # print(out.shape)
            # print(f"shape of out: {out.shape}; shape of label: {label.shape}")
            # return 0
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        # print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def mlp_evaluate(model, test_loader): 
    outputs = []
    results = []
    start_time = time.time()
    model.eval()
    err = 0
    for x, label in test_loader:
        input = x.to(device)
        output = model(input)
        result = torch.argmax(output, dim=1)
        results.append(result)
        outputs.append(output)
        err += torch.abs(result.to(device) - label.to(device)).sum()
    accuracy = 1 - err/len(test_loader)
    return accuracy, outputs, results


mlp_model = mlp_train(train_loader, len(vocab), 0.01, hidden_dim=100 ,EPOCHS=20)


train_accuracy, train_outputs, train_results = mlp_evaluate(mlp_model, train_loader)
test_accuracy, train_outputs, train_results = mlp_evaluate(mlp_model, test_loader)


print(train_accuracy)
print(test_accuracy)



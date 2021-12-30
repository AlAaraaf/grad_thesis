from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import pandas as pd

MAX_LENGTH = 1024
PADDING = '[PAD]'
BATCH_SIZE = 64

def read_data(filepath = './data/train_wordseq.csv'):
    dataset = pd.read_csv(filepath, encoding='utf-8', sep = '|')
    dataset = dataset[['input_sentence','action_cause']]
    dataset['wordset'] = dataset['input_sentence'].apply(lambda x: x.split(' '))
    batch_input, batch_length, batch_target, labelmap, word2idx = make_data(dataset['wordset'].tolist(), 
                                                    dataset['action_cause'].tolist())
    batch_input = torch.LongTensor(batch_input)
    batch_target = torch.LongTensor(batch_target)
    x_train, x_test, y_train, y_test = train_test_split(batch_input, batch_target, test_size=0.2, random_state=42)
    train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    dataset = data.TensorDataset(batch_input, batch_target)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = len(word2idx)
    return train_loader, test_loader, labelmap, vocab_size, word2idx[PADDING]

def make_data(sentences, labels):
    vocab = list(set(sum(sentences, [])))
    vocab.append(PADDING)
    labelset = list(set(labels))
    word2idx = {word: idx for idx,word in enumerate(vocab)}
    labelmap = {label: idx for idx, label in enumerate(labelset)}
    input = []
    target =[]
    input_length = []
    for sentence, label in zip(sentences, labels):
        sent_length = len(sentence)
        if sent_length < MAX_LENGTH:
            sentence = sentence + [PADDING]*(MAX_LENGTH - sent_length)
        elif sent_length > MAX_LENGTH:
            # 句子过长时拼接头尾
            sentence = sentence[:MAX_LENGTH/2] + sentence[sent_length - MAX_LENGTH / 2:]
        
        input.append([word2idx[item] for item in sentence])
        target.append(labelmap[label])
        input_length.append(min(sent_length, MAX_LENGTH))
    
    return input, input_length, target, labelmap, word2idx
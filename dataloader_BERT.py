from torch.utils.data import Dataset
import random
import torch
from torch.utils.data import DataLoader
import re
from collections import Counter

class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab_stoi, vocab_itos, seq_len):
        super().__init__()
        self.vocab_stoi = vocab_stoi #string to integer
        self.vocab_itos = vocab_itos # integer to string

        self.seq_len = seq_len
        self.ignore_idx = self.vocab_stoi["<ignore>"]
        self.oov_idx = self.vocab_stoi["<oov>"]
        self.mask_idx = self.vocab_stoi["<mask>"]

        self.sentences= [[self.vocab_stoi.get(word,self.oov_idx) for word in sentence] for sentence in sentences]
    def __getitem__(self, index):
        sentence = self.sentences[index]
        sentence = [(word, self.ignore_idx) if random.random()>=0.15 else (self.mask_idx,word) for word in sentence]
        sentence_input = [word[0] for word in sentence]
        sentence_target = [word[1] for word in sentence]
        return torch.tensor(sentence_input, dtype=torch.int64), torch.tensor(sentence_target, dtype= torch.int64)

    def __len__(self):
        return len(self.sentences)

def get_sentence_dataloader(filename, vocab_size,seq_len, batch_size):
    # preprocess the data
    with open(filename, "r") as f:
        data = f.read()
        data = re.sub("\n"," ", data)
        escape_character = re.escape(',?;.:/*!+-()[]{}"\'&')
        data = re.sub(f"[{escape_character}]"," \g<0> ",data)
        data = data.split(" ")
        data= [word.strip() for word in data if len(word.strip())>0]

    #creation of vocab
    count = Counter(data)
    
    vocab = [word[0] for word in count.most_common(vocab_size-3)]
    vocab = vocab + ["<ignore>","<oov>","<mask>"]

    print(f"total number of unique words are {len(count)}, taking most common {vocab_size-3} words along with 3 special tokens [ignore, oov, mask]")
    
    vocab_stoi = {word:i for i, word in enumerate(vocab)} #string to integer
    vocab_itos = {i:word for word, i in vocab_stoi.items()} # integer to string


    with open("vocab.txt", "w") as f:
        f.write("\n".join(vocab))
    

    start = 0
    end = seq_len
    sentences = []
    while end<len(data):
        sentences.append(data[start:end])
        start+=seq_len
        end = start+seq_len

    train_size = int(0.8 * len(sentences))
    train_data = sentences[:train_size]
    test_data = sentences[train_size:]

    train_dataset = SentenceDataset(train_data, vocab_stoi, vocab_itos, seq_len)
    test_dataset  = SentenceDataset(test_data,  vocab_stoi, vocab_itos, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2 )
    return train_loader, test_loader, vocab_stoi, vocab_itos
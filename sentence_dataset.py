from torch.utils.data import Dataset
import random
import torch
from torch.utils.data import DataLoader
import re
from collections import Counter

class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len):
        super().__init__()
        
        self.vocab = vocab + ["<ignore>","<oov>","<mask>"] # ignore, out of vocab, mask
        self.vocab_stoi = {word:i for i, word in enumerate(self.vocab)} #string to integer
        self.vocab_itos = {i:word for word, i in self.vocab_stoi.items()} # integer to string

        self.seq_len = seq_len
        self.ignore_idx = self.vocab_stoi["<ignore>"]
        self.oov_idx = self.vocab_stoi["<oov>"]
        self.mask_idx = self.vocab_stoi["<mask>"]

        self.sentences= [[self.vocab_stoi.get(word,self.oov_idx) for word in sentence.split(" ")] for sentence in sentences]
    def __getitem__(self, index):
        sentence = self.sentences[index]
        while len(sentence)< self.seq_len:
            sentence += self.sentences[(index+1)%len(self.sentences)]
            index+=1
        sentence = sentence[:self.seq_len]
        sentence = [(word, self.ignore_idx) if random.random()>=0.5 else (self.mask_idx,word) for word in sentence]
        sentence_input = [word[0] for word in sentence]
        sentence_target = [word[1] for word in sentence]
        return torch.tensor(sentence_input, dtype=torch.int64), torch.tensor(sentence_target, dtype= torch.int64)

    def __len__(self):
        return len(self.sentences)

def get_sentence_dataloader(filename):
    # preprocess the data
    with open(filename, "r") as f:
        data = f.read().splitlines()
        data = [sentence.strip() for sentence in data]
        data = [sentence for sentence in data if len(sentence)>0]
        escape_character = re.escape(',?;.:/*!+-()[]{}"\'&')
        data = [re.sub(f"[{escape_character}]"," \g<0> ",sentence) for sentence in data]

    #creation of vocab
    
    vocab_size = 40000
    words = [word for sentence in data for word in sentence.split(" ")]
    count = Counter(words)
    vocab = [word[0] for word in count.most_common(vocab_size)]
    with open("vocab.txt", "w") as f:
        f.write("\n".join(vocab))


    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    sequence_length = 160
    train_dataset = SentenceDataset(train_data, vocab, sequence_length)
    test_dataset = SentenceDataset(test_data, vocab, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2 )
    return train_loader, test_loader
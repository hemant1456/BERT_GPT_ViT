from torch.utils.data import Dataset
import random

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
        sentence_target = sentence[:self.seq_len]
        sentence_input = [word if random.random()>=0.1 else self.mask_idx for word in sentence_target]
        return sentence_input, sentence_target

    def __len__(self):
        return len(self.sentences)
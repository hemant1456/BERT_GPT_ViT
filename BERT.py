import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from lightning import Trainer


from attention_block import MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock
from embedding import InputEmbedding, PositionalEmbedding
from dataloader_BERT import get_sentence_dataloader


import random


class BERT(L.LightningModule):
    def __init__(self, vocab_stoi,vocab_itos, d_model, num_blocks, vocab_size, dropout, seq_len):
        super().__init__()
        self.input_embed = InputEmbedding(d_model, vocab_size)
        self.pos_embed = PositionalEmbedding(d_model, seq_len, dropout)
        self.encoders = nn.Sequential(*[EncoderBlock(d_model=d_model) for _ in range(num_blocks)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.vocab_stoi = vocab_stoi
        self.vocab_itos = vocab_itos
        self.ignore_idx = vocab_stoi["<ignore>"]
        self.oov_idx    = vocab_stoi["<oov>"]
        self.mask_idx   = vocab_stoi["<mask>"]
        self.vocab_size = vocab_size
    def forward(self,x):
        x = self.input_embed(x)
        x = self.pos_embed(x)
        x = self.encoders(x)
        x = self.linear(x)
        return x
    def training_step(self, batch, batch_idx):
        enc_input, enc_output = batch
        output= self(enc_input)
        loss = F.cross_entropy(output.view(-1,self.vocab_size), enc_output.view(-1), ignore_index=self.ignore_idx)
        self.log("loss", loss, on_step=True, prog_bar=True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    def validation_step(self, batch, batch_idx):
        enc_input, enc_output = batch
        output= self(enc_input)
        prediction = output.argmax(dim=2)
        with open("validation_log.txt", "a") as f:
            idxes = [random.randint(0, 15) for _ in range(1)]
            for enc, pred in zip(enc_input[idxes], prediction[idxes]):
                f.write(" ".join([self.vocab_itos[idx.item()] for idx in enc]))
                f.write("\n")
                enc[enc==self.mask_idx] = pred[enc==self.mask_idx]
                f.write(" ".join([self.vocab_itos[idx.item()] for idx in enc]))
                f.write("\n\n")

if __name__=="__main__":
    vocab_size = 25000
    sequence_length = 64
    batch_size = 128
    train_loader, test_loader, vocab_stoi, vocab_itos = get_sentence_dataloader("training.txt", vocab_size, sequence_length,batch_size)

    trainer = Trainer(max_epochs=100, limit_val_batches=2)
    model = BERT(vocab_stoi, vocab_itos,  512, 6, vocab_size, 0.1, sequence_length)
    trainer.fit(model, train_loader, test_loader)

import torch
from embedding import PositionalEmbedding
import torch.nn as nn
from einops import rearrange
from attention_block import EncoderBlock
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ViT(L.LightningModule):
    def __init__(self, num_encoder_blocks=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, 8, 8, 4)
        self.class_token = nn.Parameter(torch.ones(1,768).unsqueeze(0),requires_grad=True)
        self.pos_embed = PositionalEmbedding(768, 26, 0.1)
        self.encoder = nn.Sequential(*[EncoderBlock() for _ in range(num_encoder_blocks)])
        self.classifier = nn.Linear(768,10)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
    def forward(self,x):
        x = self.patch_embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = torch.cat((self.class_token.expand(x.size(0),1,x.size(-1)),x), dim=1)
        x = self.pos_embed(x)
        x = self.encoder(x)
        x = x[:,0,:] # take the classifier token row
        x = self.classifier(x)
        return x
    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        pred = output.argmax(dim=1)
        loss = F.cross_entropy(output, labels)
        accuracy = self.accuracy(pred,labels)
        self.log("training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("training_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True) 
        return loss
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        pred = output.argmax(dim=1)
        loss = F.cross_entropy(output, labels)
        accuracy = self.accuracy(pred,labels)
        self.log("validation_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("validation_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True) 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


from dataloader_vit import get_dataloader
train_loader, test_loader = get_dataloader()


from lightning import Trainer
vit_model = ViT()
trainer = Trainer(max_epochs=80, limit_val_batches=10)
torch.set_float32_matmul_precision("medium")
trainer.fit(vit_model, train_loader, test_loader)
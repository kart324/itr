import time
import torch
import numpy as np
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
import pytorch_lightning as pl
from model import TranslationModel

    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)


def load_model(filepath):
    pass

def get_opts(model,lr,train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=lr)
    return optimizer,scheduler     

class pyLight(pl.LightningModule):
    def __init__(self, config, model,train_loader):
        super().__init__()
        self.model = model
        self.config = config
        self.train_loader = train_loader
    def forward(self, encoder_input_ids, decoder_input_ids):
        encoder_hidden_states = self.model.encoder(encoder_input_ids)[0]
        loss,logits = self.model.decoder(decoder_input_ids,encoder_hidden_states=encoder_hidden_states,masked_lm_labels=decoder_input_ids)
        return loss,logits
    #def forward(self, x):
    #    return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x = batch[0].to(device)
        y = batch[1].to(device)
        loss,_ = self.forward(x, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x = batch[0].to(device)
        y = batch[1].to(device)
        loss, logits = self.forward(x, y)
        #y_hat = self.forward(x)
        #preds = torch.argmax(y_hat, dim=1)
        logits = logits.detach().cpu().numpy()
        label_ids = y.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=2).flatten()
        labels_flat = label_ids.flatten()
        return {'val_loss': loss, 'correct': torch.from_numpy(np.equal(pred_flat,labels_flat)).float()}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.cat([x['correct'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        # REQUIRED
        optimizer, scheduler = get_opts(self.model,self.config.lr,self.train_loader)
        return [optimizer], [scheduler]

    

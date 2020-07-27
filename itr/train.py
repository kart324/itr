from time import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_util import pyLight
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers





def preproc_data():
    from data import split_data
    split_data('/content/drive/My Drive/OffNote/itr-master/data/hin-eng/hin.txt', '/content/drive/My Drive/OffNote/itr-master/data/hin-eng')


from data import IndicDataset, PadSequence
import model as M


def gen_model_loaders(config):
    model, tokenizers = M.build_model(config)

    pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True), 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False), 
                           batch_size=config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    return model, tokenizers, train_loader, eval_loader



from config import replace, preEnc, preEncDec

def main():
    rconf = preEncDec
    model, tokenizers, train_loader, eval_loader = gen_model_loaders(rconf)
    writer = SummaryWriter(rconf.log_dir)
    #train_losses, val_losses, val_accs = run_train(rconf, model, train_loader, eval_loader, writer)
    tb_logger = pl_loggers.TensorBoardLogger('/content/drive/My Drive/Offnote-New/itr-master/logs/pretrained-enc-dec')
    LightTrain = pyLight(rconf,model,train_loader)
    trainer = pl.Trainer(max_epochs=20,logger=tb_logger)
    trainer.fit(LightTrain,train_dataloader=train_loader,val_dataloaders=eval_loader)
    model.save(model,rconf.model_output_dirs)

if __name__ == '__main__':
    preproc_data()
    main()









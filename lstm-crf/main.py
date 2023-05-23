import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from pytorch_lightning.loggers import TensorBoardLogger

from models import NERModel 
from readdata import NERDataset, collate_wapper
from all_utils import Config

def run_with(config: Config):
    seed_everything(config.seed)

    # Create a BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    # Load the training, validation, and test datasets
    train_dataset = NERDataset(dir=config.data_dir+config.train_file, label2id=config.label2id, language=config.language, tokenizer=tokenizer)
    val_dataset = NERDataset(dir=config.data_dir+config.dev_file, label2id=config.label2id, language=config.language, tokenizer=tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_wapper(pad_idx=tokenizer.pad_token_id),
                            shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_wapper(pad_idx=tokenizer.pad_token_id),
                            shuffle=False, num_workers=8)

    # Initialize our model
    model = NERModel(config)

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        devices="auto",
        logger=TensorBoardLogger(save_dir='./logs/', name=config.model_name),
        callbacks=[
            EarlyStopping(monitor='val_f1', mode='max', patience=5),
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                filename=config.model_name,
                monitor='val_f1',
                save_top_k=1,
                mode='max',
            ),
        ],
        fast_dev_run=False
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


def test_with(config: Config):
    # Load the best model
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    test_dataset = NERDataset(dir=config.data_dir + config.test_file, label2id=config.label2id, language=config.language, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_wapper(pad_idx=tokenizer.pad_token_id),
                            shuffle=False, num_workers=8)

    model = NERModel.load_from_checkpoint(config.ckpt_dir + config.model_name + '.ckpt', config=config)

    # Evaluate the model on test data
    model.eval()
    model.cuda()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, word_ids, mask = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            word_ids = tuple(t.cuda() for t in word_ids)
            mask = mask.cuda()
            predictions = model(input_ids, word_ids, mask)
            all_predictions.append(predictions.flatten())
            all_labels.append(labels.flatten())

        # Compute the F1 score
        y_true = torch.cat(all_labels, dim=0).cpu().tolist()
        y_pred = torch.cat(all_predictions, dim=0).cpu().tolist()

    y_true = [config.id2label[it] for it in y_true]
    y_pred = [config.id2label[it] for it in y_pred]
    print(classification_report(y_true, y_pred, labels=config.sort_labels[1:], digits=4))

if __name__ == '__main__':
    config = Config.get_config('./lstm-crf/eng.yaml')
    run_with(config)
    test_with(config)
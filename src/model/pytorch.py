import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR  #
from torch.utils.data import TensorDataset, \
    DataLoader  # Own stuff set of data quality checks, tensor shapes might be different dataloader loads the tensor,
import pytorch_lightning as pl  #
from pytorch_lightning.callbacks.early_stopping import \
    EarlyStopping  # early stop when you reach optimum loss, 3 times in a row gradient descent
from pytorch_lightning.callbacks import LearningRateMonitor  # delta (loss / accuracy)
from pytorch_lightning.loggers import MLFlowLogger  # Model tracking
import torch
import torch.nn as nn
from transformers import DistilBertModel


class MultiClassClassifier(pl.LightningModule):
    def __init__(self, seed, output_dim, input_dim=4, hidden_dim=5, learning_rate=1e-3, max_learning_rate=0.1, total_steps=100):
        # lower than epoch higher than iterations
        super().__init__()
        self.save_hyperparameters()  # save_hyperparameters
        pl.seed_everything(seed)
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768,output_dim)

    def forward(self, batch):
        # input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
        # input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
        embeddings = self.model(batch['input_ids'], attention_mask=batch['attention_mask'],return_dict=False)[0]
        pooler = self.pre_classifier(embeddings)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = OneCycleLR(optimizer, self.hparams.max_learning_rate, self.hparams.total_steps)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = F.nll_loss(logits, batch['labels'])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = F.nll_loss(logits, batch['labels'])
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = F.nll_loss(logits, batch['labels'])
        self.log("test_loss", loss)

    def predict(self, x):
        # TODO this will be changed
        self.eval()
        logits = self.forward(x)
        self.train()
        return torch.argmax(logits, dim=1).detach().numpy()

    # def df_to_tensor(self, df, target_col=None, format=np.float32):
    #     if target_col is not None:
    #         feature_cols = [col for col in df.columns if col != target_col]
    #         tensor = TensorDataset(
    #             torch.tensor(df[feature_cols].values.astype(format)),
    #             torch.tensor(df[target_col].values),
    #         )
    #     else:
    #         tensor = torch.tensor(df.values.astype(format))
    #     return tensor

    def tensor_to_loader(self, tensor, batch_size, num_workers, shuffle=True):
        loader = DataLoader(dataset=tensor, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return loader

    def setup_trainer(self, experiment, run_id, max_epochs):
        mlf_logger = MLFlowLogger(experiment_name=experiment)
        mlf_logger._run_id = run_id
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="val_loss"), LearningRateMonitor(logging_interval="epoch")],
            logger=mlf_logger,
        )

        return trainer


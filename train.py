from torch.utils.data import DataLoader
import pytorch_lightning
from model import AnimalModel
from dataloader import create_dataloaders

import torch

torch.manual_seed(0)

# Model training loop
if __name__ == "__main__":

    train_set, val_set = create_dataloaders()

    early_stopping_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0.005,
        patience=15,
        verbose=True,
    )

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(last=True)

    trainer = pytorch_lightning.Trainer(
        gpus=1,
        max_epochs=100,
        precision=16,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    model = AnimalModel()

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_set, shuffle=True, num_workers=4, batch_size=256
        ),
        val_dataloaders=DataLoader(
            val_set, shuffle=False, num_workers=4, batch_size=256
        ),
    )

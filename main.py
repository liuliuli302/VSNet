from lightning import Trainer



def train():
    model = None
    datamodule = None
    trainer = Trainer(max_epochs=150,
                      enable_checkpointing=False)
    trainer.fit(model, datamodule=datamodule)



import omegaconf
import hydra
import pytorch_lightning as pl
from src.pl_modules.steps_data_modules import STEPSDataModule

def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)
    
    hydra.utils.log.info(f"Data Module")

    # data module
    pl_data_module = STEPSDataModule(conf)
    
    hydra.utils.log.info(f"STEPS Module")
    # model module
    pl_module = hydra.utils.instantiate(
        conf.model
    )

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            filename="{epoch:02d}-{" + conf.train.callbacks_monitor + ":.2f}",
        )
        callbacks_store.append(model_checkpoint)

    # trainer
    trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, 
        callbacks=callbacks_store,
        logger = False)

    hydra.utils.log.info(f"Start Training")
    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
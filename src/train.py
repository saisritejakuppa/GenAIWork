from logzero import logger
from src.operation.model_train.gan_training import GanTraining


class Train:
    def __init__(self, config):
        self.config = config

    def train(self):
        if self.config["model"]["model"] == "gan":
            logger.debug("Training a Gan Arch")
            GanTraining(self.config).train()
            pass

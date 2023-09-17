from logzero import logger
import torch
from src.genai.model.Generators.gen import Generator
from src.genai.model.discriminators.disc import Discriminator
from src.genai.dataoperations.dataoperations import DataLoadingOperations
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import torchvision
import os


class TestConfig:
    def __init__(self) -> None:
        pass


class GanTesting:
    def __init__(self, config) -> None:
        self.config = config

    def test(self):
        model_info = self.config["gan"]

        if model_info["name"] == "simple_gan":
            self.trainconfig = TestConfig()
            self.trainconfig.data = 0
            self.trainconfig.device = self.config["operation"]["device"]
            self.trainconfig.disc = Discriminator(self.config["gan"]["image_dim"])
            self.trainconfig.gen = Generator(self.config["gan"]["z_dim"], self.config["gan"]["image_dim"])

            # print(self.trainconfig.gen)
            # print(self.trainconfig.disc)

            logger.debug("The Generator and Discriminator are built.")

            tfms = DataLoadingOperations.get_tfs()
            data = DataLoadingOperations.get_mnist(tfms)
            self.trainconfig.dataloader = DataLoadingOperations.dataloading(data)

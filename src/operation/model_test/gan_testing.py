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
            wandb_status = self.config["operation"]["wandb_flag"]

            # print(self.trainconfig.gen)
            # print(self.trainconfig.disc)

            logger.debug("The Generator and Discriminator are built.")

            tfms = DataLoadingOperations.get_tfs()
            data = DataLoadingOperations.get_mnist(tfms)
            self.trainconfig.dataloader = DataLoadingOperations.dataloading(data)

            model_weights = torch.load("./output/op.pt")

            self.trainconfig.gen.load_state_dict(model_weights["model_gen"])
            self.trainconfig.disc.load_state_dict(model_weights["model_disc"])
            logger.info("loaded the pretrained weights")

            batch_size = self.config["gan"]["batch_size"]
            z_dim = self.config["gan"]["z_dim"]
            device = self.config["operation"]["device"]
            noise = torch.randn((batch_size, z_dim)).to(device)

            gen = self.trainconfig.gen.to(device)

            if wandb_status:
                wandb.init(project="gan_testing", config=self.config)

            with torch.no_grad():
                for _ in range(10):
                    fake = gen(noise).reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_fake = wandb.Image(
                        img_grid_fake,
                    )

                    wandb.log({"img_grid_fake": img_grid_fake})

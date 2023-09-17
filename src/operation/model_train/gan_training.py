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


class TrainConfig:
    def __init__(self) -> None:
        pass


class GanTraining:
    def __init__(self, config) -> None:
        self.config = config

    def optimizer(self, params, model):
        if params["name"] == "adam":
            return optim.Adam(model.parameters(), lr=params["lr"])

    def criterion(self, params):
        if params["name"] == "bceloss":
            return nn.BCELoss()

    def train(self):
        model_info = self.config["gan"]

        if model_info["name"] == "simple_gan":
            self.trainconfig = TrainConfig()
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

            logger.debug("The dataloader is loaded.")

            gan_params = self.config["gan"]["gen"]["opt"]
            self.trainconfig.gen_opt = self.optimizer(gan_params, self.trainconfig.gen)

            disc_params = self.config["gan"]["disc"]["opt"]
            self.trainconfig.disc_opt = self.optimizer(disc_params, self.trainconfig.disc)

            logger.debug("The optimizers are init.")

            criterion_params = self.config["gan"]["criterion"]
            self.trainconfig.criterion = self.criterion(criterion_params)

            logger.debug("The criterion is set.")

            self.trainconfig.device = self.config["operation"]["device"]
            self.trainconfig.z_dim = self.config["gan"]["z_dim"]
            self.trainconfig.batch_size = self.config["gan"]["batch_size"]
            self.trainconfig.num_epochs = self.config["gan"]["num_epochs"]

            self.trainconfig.wandb_status = self.config["operation"]["wandb_flag"]
            freq = self.config["operation"]["wandb_model_freq"]
            if self.trainconfig.wandb_status:
                wandb.init(project="gan", config=self.config)
                wandb.watch(self.trainconfig.disc, log_freq=freq, log="all")
                wandb.watch(self.trainconfig.gen, log_freq=freq, log="all")

            self.simple_gan_train(self.trainconfig)

    def simple_gan_train(self, config):
        logger.debug("training a simple gan")

        # make sure the below are there configured in the config dic
        # before training.
        device = self.trainconfig.device
        z_dim = self.trainconfig.z_dim
        batch_size = self.trainconfig.batch_size
        num_epochs = self.trainconfig.num_epochs

        disc = config.disc
        gen = config.gen

        loader = config.dataloader
        opt_disc = config.disc_opt
        opt_gen = config.gen_opt

        criterion = config.criterion
        wandb_status = config.wandb_status

        # convert models to device
        disc = disc.to(device)
        gen = gen.to(device)

        fixed_noise = torch.randn((batch_size, z_dim)).to(device)
        step = 0

        for epoch in range(num_epochs):
            for batch_idx, (real, _) in enumerate(loader):
                real = real.view(-1, 784).to(device)
                batch_size = real.shape[0]

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise)
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output))
                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                if wandb_status:
                    wandb_logs = {"lossD": lossD, "lossG": lossG, "epoch": epoch}
                    wandb.log(wandb_logs)

                if batch_idx == 0:
                    print("Epoch :", epoch)
                    print("Disc Loss :", lossD)
                    print("Gen Loss :", lossG)

                    with torch.no_grad():
                        fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                        data = real.reshape(-1, 1, 28, 28)
                        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                        img_grid_fake = wandb.Image(
                            img_grid_fake,
                        )

                        wandb.log({"img_grid_fake": img_grid_fake})

                        img_grid_real = wandb.Image(
                            img_grid_real,
                        )

                        wandb.log({"img_grid_real": img_grid_real})

                    step += 1

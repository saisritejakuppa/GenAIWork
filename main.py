import toml
from src.train import Train
from src.infer import Infer
from src.test import Test
from logzero import logger


def main(config):
    if config["operation"]["operation"] == "train":
        logger.info("Operation: Training")
        Train(config).train()

    if config["operation"]["operation"] == "test":
        logger.info("Operation: Testing")
        Test(config).test()

    if config["operation"]["operation"] == "infer":
        logger.info("Operation: Inference")
        Infer(config).infer()

    return


if __name__ == "__main__":
    toml_data = toml.load("config.toml")
    main(toml_data)

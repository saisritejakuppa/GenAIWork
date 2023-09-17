from logzero import logger
from src.operation.model_test.gan_testing import GanTesting


class Test:
    def __init__(self, config):
        self.config = config

    def test(self):
        if self.config["model"]["model"] == "gan":
            logger.debug("Training a Gan Arch")
            GanTesting(self.config).test()
            pass

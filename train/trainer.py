from datetime import datetime

from utils import Logger
from .dp import process
from .test import test


class Trainer(object):
    def __init__(self, para, config_path=None):
        self.para = para
        self.config_path = config_path

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para, self.config_path)
        logger.record_para()

        # training
        if not self.para.test_only:
            process(self.para)

        # test
        test(self.para, logger)

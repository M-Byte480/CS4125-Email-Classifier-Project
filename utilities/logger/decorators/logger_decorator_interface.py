from abc import ABC, abstractmethod

from utilities.logger.abstract_logger import AbstractLogger


class ILoggerDecorator(AbstractLogger, ABC):
    def __init__(self, logger: AbstractLogger):
        super().__init__()
        self.logger = logger

    @abstractmethod
    def log(self, message):
        pass

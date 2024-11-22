from utilities.logger.decorators.logger_decorator_interface import ILoggerDecorator


class ItalicsDecorator(ILoggerDecorator):
    ITALICS = "\x1B[3m"
    END_ITALICS = "\x1B[0m"

    def __init__(self, logger):
        super().__init__(logger)

    def log(self, message=""):
        self.logger.log(f"{self.ITALICS}{message}{self.END_ITALICS}")
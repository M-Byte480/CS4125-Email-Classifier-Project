from utilities.logger.decorators.logger_decorator_interface import ILoggerDecorator


class BoldDecorator(ILoggerDecorator):
    BOLD = "\u001b[1m"
    END_BOLD = "\033[0m"

    def __init__(self, logger):
        super().__init__(logger)

    def log(self, message=""):
        self.logger.log(f"{self.BOLD}{message}{self.END_BOLD}")
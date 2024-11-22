from utilities.logger.abstract_logger import AbstractLogger


class WarningLogger(AbstractLogger):
    def __init__(self, running_location):
        super().__init__()
        self.base_message = "[WARNING]"
        self.running_location = running_location

    def log(self, message):
        print(self.base_message, f"[{self.running_location}]",  message)
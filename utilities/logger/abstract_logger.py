from abc import ABC, abstractmethod

class AbstractLogger(ABC):
    __slots__ = ["base_message", "running_location"]

    def __init__(self):
        self.base_message = "[BASE LOG]"
        self.running_location = "[AbstractLogger]"
        ...

    @abstractmethod
    def log(self, message: str):
        pass

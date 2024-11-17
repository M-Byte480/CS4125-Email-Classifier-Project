from abc import ABC, abstractmethod

class EmailClassificationObserver(ABC):
    @abstractmethod
    def update(self, classification: str) -> None:
        """This method is called when the subject notifies its subscribers"""
        pass
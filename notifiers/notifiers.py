from abc import ABC, abstractmethod

from structs.objects import Email


class Observer(ABC):
    @abstractmethod
    def update(self, email: Email, classification):
        pass



class EmailClassifier:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def classify_email(self, email) -> None:
        classification = "spam" if "spam" in email.lower() else "not spam"
        self.notify_observers(email, classification)

    def notify_observers(self, email, classification) -> None:
        for observer in self._observers:
            observer.update(email, classification)

# Concrete Observer Example
class LoggingService(Observer):
    def update(self, email, classification):
        print(f"LoggingService: Email '{email}' classified as '{classification}'")

class NotificationService(Observer):
    def update(self, email, classification):
        print(f"NotificationService: Sending notification for email '{email}' classified as '{classification}'")
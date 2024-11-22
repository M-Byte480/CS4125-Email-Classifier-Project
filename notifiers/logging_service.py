# Concrete Observer Example
from notifiers.observer.observer import Observer


class LoggingService(Observer):
    def update(self, email, classification):
        print(f"LoggingService: Email '{email}' classified as '{classification}'")


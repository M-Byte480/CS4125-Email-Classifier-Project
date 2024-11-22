# Concrete Observer Example
from notifiers.observer.observer import Observer


class NotificationService(Observer):
    def update(self, email, classification):
        print(f"NotificationService: Sending notification for email '{email}' classified as '{classification}'")
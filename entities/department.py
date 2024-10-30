from .notification_entities import Email, SMS, Push

class Department:
    list_of_mediums = [Email, SMS, Push]

    def __init__(self, name: str, medium_preference):
        self.name = name
        self.medium_preference = medium_preference
        self.notification_medium = None

    # todo: Add factory method
    def notify(self, subject: str, body: str):
        self.notification_medium.notify()
        pass

    def __str__(self):
        return f"{self.name}"
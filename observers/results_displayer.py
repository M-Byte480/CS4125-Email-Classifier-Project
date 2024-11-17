from typing import override

from observers.email_classification_observer import EmailClassificationObserver
from structs.objects import Email


class ResultsDisplayer(EmailClassificationObserver):
    @override
    def update(self, classification: str) -> None:
        self._display(classification)

    def _display(self, classification: str) -> None:
        """Print classification result."""
        print(f"\nEmail classification result: {classification}")
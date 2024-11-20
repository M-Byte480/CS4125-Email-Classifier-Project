from typing import override

from observers.email_classification_observer import EmailClassificationObserver

class ResultsDisplayer(EmailClassificationObserver):
    @override
    def update(self, classification: str) -> None:
        self._display(classification)

    def _display(self, classification: str) -> None:
        """Print classification result."""
        print(f"\nEmail classification result: {classification}")

class DisplayResultsCommand:
    disp : ResultsDisplayer
    classy : EmailClassificationObserver

    def __init__(self, d: ResultsDisplayer, c : EmailClassificationObserver):
        self.disp = d
        self.classy = c

    def execute(self):
        self.disp.update(self.classy)

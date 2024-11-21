from typing import override

from observers.email_classification_observer import EmailClassificationObserver

class ResultsDisplayer(EmailClassificationObserver):
    @override
    def update(self, ts, ic, classification: str) -> None:
        self._display(ts, ic, classification)

    def _display(self, ts, ic, classification: str) -> None:
        """Print classification result."""
        print(f"""Email classification result:
    Ticket summary: {ts}
    Interaction content: {ic}
    Classification: {classification}
""")

# todo: review unused class
class DisplayResultsCommand:
    disp : ResultsDisplayer
    classy : EmailClassificationObserver

    def __init__(self, d: ResultsDisplayer, c : EmailClassificationObserver):
        self.disp = d
        self.classy = c

    def execute(self):
        self.disp.update(self.classy)

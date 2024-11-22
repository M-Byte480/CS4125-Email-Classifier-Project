from typing import override

from observers.email_classification_observer import EmailClassificationObserver
from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.indentation_decorator import IndentationDecorator
from utilities.logger.info_logger import InfoLogger


class ResultsDisplayer(EmailClassificationObserver):
    logger: AbstractLogger = InfoLogger("ResultsDisplayer")
    @override
    def update(self, ts, ic, classification: str) -> None:
        self._display(ts, ic, classification)

    def _display(self, ts, ic, classification: str) -> None:
        indentation_logger = IndentationDecorator(ResultsDisplayer.logger, "[ResultsDisplayer]")
        """Print classification result."""
        ResultsDisplayer.logger.log(f"Email classification result:")
        indentation_logger.log(f"Ticket summary: {ts}")
        indentation_logger.log(f"Interaction content: {ic}")
        indentation_logger.log(f"Classification: {classification}")
        indentation_logger.log("=" * 75)

# todo: review unused class
class DisplayResultsCommand:
    disp : ResultsDisplayer
    classy : EmailClassificationObserver

    def __init__(self, d: ResultsDisplayer, c : EmailClassificationObserver):
        self.disp = d
        self.classy = c

    def execute(self):
        self.disp.update(self.classy)


from observers.email_classification_observer import EmailClassificationObserver
from utilities.logger.decorators.indentation_decorator import IndentationDecorator
from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class ResultsDisplayer(EmailClassificationObserver):
    info_logger = InfoLogger()

    def update(self, ts, ic, classification: str) -> None:
        self._display(ts, ic, classification)

    def _display(self, ts, ic, classification: str) -> None:
        """Print classification result."""
        display_logger = PrefixLogger(ResultsDisplayer.info_logger, "ResultsDisplayer")
        display_logger = IndentationDecorator(display_logger)

        display_logger.log(f"Email classification result:")
        display_logger.log(f"Ticket summary: {ts}")
        display_logger.log(f"Interaction content: {ic}")
        display_logger.log(f"Classification: {classification}")
        display_logger.log("=" * 75)
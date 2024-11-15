import random

from modelling.data_model import ClassificationContextFactory
from observers.results_displayer import ResultsDisplayer
from observers.statistics_collector import StatisticsCollector
from structs.objects import Email


def main():
    # Create an email classifier
    classifier_factory = ClassificationContextFactory()
    classifier = classifier_factory.create_context("random_forest")

    # Create our observers
    results_displayer = ResultsDisplayer()
    statistics_collector = StatisticsCollector()

    # Subscribe the observers to the email classifier
    classifier.add_observer(results_displayer)
    classifier.add_observer(statistics_collector)

    # Create mock emails with specified classifications
    email1 = Email(i=1, t=1, sen="", r="", sub="", bt="", c="AppGallery")
    email2 = Email(i=1, t=1, sen="", r="", sub="", bt="", c="In-App Purchase")
    email3 = Email(i=1, t=1, sen="", r="", sub="", bt="", c="Incorrect label")

    # Classify the emails a random number of times
    num = random.randint(1, 10)
    for i in range(num):
        classifier.classify_email(email1)

    num = random.randint(1, 10)
    for i in range(num):
        classifier.classify_email(email2)

    num = random.randint(1, 5)
    for i in range(num):
        classifier.classify_email(email3)

    # Display the statistical report
    statistics_collector.display_stats()

if __name__ == "__main__":
    main()
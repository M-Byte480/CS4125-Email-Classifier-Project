import numpy as np

from typing import override

from observers.email_classification_observer import EmailClassificationObserver

#StatisticsCollector keeps track of classification statistics for an email classifier.
class StatisticsCollector(EmailClassificationObserver):
    _stats: dict[str, int]

    def __init__(self):
        self._stats = {"AppGallery": 0, "In-App Purchase": 0, "Unexpected Label": 0}
        # todo: check back here after we decide how to deal with labels

    @override
    def update(self, classification: str) -> None:
        self._update_stats(classification)

    def display_stats(self) -> None:
        """Print out a report of collected statistics."""
        total_stats_collected = sum(self._stats.values())
        print(f"Statistics report:")

        for key, value in self._stats.items():
            percentage = np.round(value / total_stats_collected * 100, 2)
            print(f" - {key}: {value} | {percentage}%")

        print(f"Total stats collected: {total_stats_collected}")


    def _update_stats(self, classification: str) -> None:
        """Update statistics."""
        if classification in self._stats.keys():
            self._stats[classification] += 1
        else:
            self._stats["Unexpected Label"] += 1
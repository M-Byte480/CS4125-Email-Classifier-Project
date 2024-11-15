from typing import override

from observers.email_classification_observer import EmailClassificationObserver
import numpy as np

class StatisticsCollector(EmailClassificationObserver):
    _stats: dict[str, int]

    def __init__(self):
        self._stats = {"AppGallery": 0, "In-App Purchase": 0, "Unexpected Label": 0}

    @override
    def update(self, classification: str) -> None:
        self._update_stats(classification)

    def display_stats(self) -> None:
        """Print out a report of collected statistics."""
        total_stats_collected = np.sum(self._stats)
        print(f"\nStatistics report:")

        for key, value in self._stats.items():
            percentage = np.round(value / total_stats_collected * 100, 2)
            print(f"\n - {key}: {value} | {percentage}%")

        print(f"\nTotal stats collected: {total_stats_collected}")


    def _update_stats(self, classification: str) -> None:
        """Update statistics."""
        if classification in self._stats.keys():
            self._stats[classification] += 1
        else:
            self._stats["Unexpected Label"] += 1
import numpy as np

from typing import override

from observers.email_classification_observer import EmailClassificationObserver

# StatisticsCollector keeps track of classification statistics for an email classifier.
class StatisticsCollector(EmailClassificationObserver):
    _label_1_stats: dict[str, int]
    _label_2_stats: dict[str, int]
    _label_3_stats: dict[str, int]
    _label_4_stats: dict[str, int]
    _unclassified: int
    _total_classifications: int

    def __init__(self):
        self._label_1_stats = {"AppGallery &amp; Games ": 0, "In-App Purchase ": 0}
        self._label_2_stats = {"Others": 0, "Problem/Fault": 0, "Suggestion": 0}
        self._label_3_stats = {"AppGallery-Install/Upgrade": 0, "AppGallery-Use": 0, "Third Party APPs": 0,
                               "VIP / Offers / Promotions": 0, "General": 0, "Coupon/Gifts/Points Issues": 0,
                               "Other": 0, "Payment": 0, "Payment issue": 0, "Invoice": 0}
        self._label_4_stats = {"Can't update Apps": 0, "Others ": 0, "Refund": 0, "Offers / Vouchers / Promotions": 0,
                               "Can't download Apps": 0, "Cannot connect - Server": 0, "Can't install Apps": 0,
                               "Other download/install/update issue": 0, "Personal data": 0,
                               "AppGallery not loading": 0, "Can't use or acquire": 0,
                               "UI Abnormal in Huawei AppGallery": 0, "Security issue / malware": 0,
                               "Cooperated campaign issue": 0, "Subscription cancellation": 0,
                               "Within 14 days of purchase (not product issue)": 0, "Query deduction details": 0,
                               "Payment failed": 0, "Invoice related request": 0, "Risk Control": 0}
        self._unclassified = 0
        self._total_classifications = 0

    @override
    def update(self, _, __, classification: str) -> None:
        self._update_stats(classification)

    def display_stats(self) -> None:
        """Print out a report of collected statistics."""
        total_emails_classified = self._total_classifications // 4

        print(f"Statistics report:")
        if sum(self._label_1_stats.values()) > 0:
            print("\n   Type 1:")
            for key, value in self._label_1_stats.items():
                percentage = np.round(value / sum(self._label_1_stats.values()) * 100, 2)
                print(f"    - {key.ljust(47)}: {value} | {percentage}%")

        if sum(self._label_2_stats.values()) > 0:
            print("\n   Type 2:")
            for key, value in self._label_2_stats.items():
                percentage = np.round(value / sum(self._label_2_stats.values()) * 100, 2)
                print(f"    - {key.ljust(47)}: {value} | {percentage}%")

        if sum(self._label_3_stats.values()) > 0:
            print("\n   Type 3:")
            for key, value in self._label_3_stats.items():
                percentage = np.round(value / sum(self._label_3_stats.values()) * 100, 2)
                print(f"    - {key.ljust(47)}: {value} | {percentage}%")

        if sum(self._label_4_stats.values()) > 0:
            print("\n   Type 4:")
            for key, value in self._label_4_stats.items():
                percentage = np.round(value / sum(self._label_4_stats.values()) * 100, 2)
                print(f"    - {key.ljust(47)}: {value} | {percentage}%")

        print(f"\nTotal classifications made:   {self._total_classifications}")
        print(f"Total emails classified:      {total_emails_classified}")
        print(f"Total unclassified labels:    {self._unclassified}")

    def _update_stats(self, classification: str) -> None:
        """Update statistics."""
        if classification in self._label_1_stats.keys():
            self._label_1_stats[classification] += 1
        elif classification in self._label_2_stats.keys():
            self._label_2_stats[classification] += 1
        elif classification in self._label_3_stats.keys():
            self._label_3_stats[classification] += 1
        elif classification in self._label_4_stats.keys():
            self._label_4_stats[classification] += 1
        else:
            self._unclassified += 1
        self._total_classifications += 1
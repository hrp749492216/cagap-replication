from typing import List
from sklearn.metrics import f1_score

_LABELS = ("R", "P", "C")


class ReliabilityGate:
    """GPT-5.5 W3 hard gate.

    Per-mode (Chat vs Tool) F1 >= 0.80 against gold on spot-check.
    If the gate fails, halt for review before confirmatory inference.
    """
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def check_reliability(self, y_true: List[str], y_pred: List[str]) -> bool:
        """Returns True if F1 score is above threshold."""
        if not y_true and not y_pred:
            return False
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return False
        observed = set(y_true) | set(y_pred)
        if not observed.issubset(_LABELS):
            return False
        labels = [label for label in _LABELS if label in observed]

        # This gate checks class-balanced label recovery against gold labels.
        # Ordinal agreement diagnostics are reported separately in the SAP.
        f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        return f1 >= self.threshold

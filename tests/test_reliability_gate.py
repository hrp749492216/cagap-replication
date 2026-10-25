from judges.reliability_gate import ReliabilityGate

def test_reliability_gate_pass():
    gate = ReliabilityGate(threshold=0.80)
    y_true = ["R", "P", "C", "R", "C"]
    y_pred = ["R", "P", "C", "R", "C"]
    assert gate.check_reliability(y_true, y_pred) is True

def test_reliability_gate_fail():
    gate = ReliabilityGate(threshold=0.80)
    y_true = ["R", "P", "C", "R", "C"]
    y_pred = ["C", "C", "R", "P", "R"] # bad match
    assert gate.check_reliability(y_true, y_pred) is False

def test_reliability_gate_empty():
    gate = ReliabilityGate()
    assert gate.check_reliability([], []) is False


def test_reliability_gate_fails_when_predictions_missing_against_gold():
    gate = ReliabilityGate()
    assert gate.check_reliability(["R", "P"], []) is False


def test_reliability_gate_fails_on_length_mismatch():
    gate = ReliabilityGate()
    assert gate.check_reliability(["R", "P"], ["R"]) is False


def test_reliability_gate_perfect_single_class_predictions_pass():
    gate = ReliabilityGate()
    assert gate.check_reliability(["R", "R"], ["R", "R"]) is True


def test_reliability_gate_fails_on_out_of_vocabulary_label():
    gate = ReliabilityGate()
    assert gate.check_reliability(["R", "P"], ["R", "X"]) is False

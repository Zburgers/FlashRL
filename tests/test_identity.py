from flashrl.identity import (
    algorithm_id,
    canonical_json,
    hyperparameter_hash,
    sha256_bytes,
)


def test_canonical_json_and_hash_ignore_mapping_order():
    first = {"learning_rate": 0.001, "gamma": 0.99}
    second = {"gamma": 0.99, "learning_rate": 0.001}
    assert canonical_json(first) == canonical_json(second)
    assert hyperparameter_hash(first) == hyperparameter_hash(second)


def test_hyperparameter_change_changes_identity():
    first = hyperparameter_hash({"learning_rate": 0.001})
    second = hyperparameter_hash({"learning_rate": 0.0001})
    assert first != second
    assert len(first) == 16


def test_sha256_bytes_is_stable():
    assert sha256_bytes(b"flashrl") == sha256_bytes(b"flashrl")
    assert len(sha256_bytes(b"flashrl")) == 64


def test_algorithm_id_describes_enabled_components():
    assert algorithm_id(False, False, False, 1) == "dqn"
    assert algorithm_id(True, False, False, 1) == "double_dqn"
    assert algorithm_id(True, True, False, 1) == "dueling_double_dqn"
    assert algorithm_id(True, True, True, 3) == "dueling_double_dqn_per_n3"


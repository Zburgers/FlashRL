from flashrl.doctor import collect_diagnostics


def test_doctor_reports_core_runtime_and_optional_features(tmp_path):
    diagnostics = collect_diagnostics(artifact_dir=tmp_path)
    assert diagnostics["flashrl_version"]
    assert diagnostics["python_version"]
    assert diagnostics["torch_version"]
    assert diagnostics["gymnasium_version"]
    assert diagnostics["artifact_directory"]["writable"] is True
    assert diagnostics["compute_device"] in {"cpu", "cuda"}
    assert set(diagnostics["optional_features"]) == {"browser", "ppo"}

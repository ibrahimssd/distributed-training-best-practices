from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_required_files_exist():
    required_files = [
        REPO_ROOT / "train_fsdp.py",
        REPO_ROOT / "train_hybrid.py",
        REPO_ROOT / "docs" / "CLUSTER_SETUP.md",
        REPO_ROOT / "docs" / "TROUBLESHOOTING.md",
        REPO_ROOT / "docs" / "BENCHMARKS.md",
        REPO_ROOT / "docs" / "BEST_PRACTICES.md",
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
    ]
    for file_path in required_files:
        assert file_path.exists(), f"Missing required file: {file_path}"


def test_new_config_templates_load():
    config_files = [
        "bert_110m_4gpu.yaml",
        "llm_1b_8gpu.yaml",
        "llm_3b_16gpu.yaml",
        "llama_7b_32gpu.yaml",
        "llama_13b_64gpu.yaml",
        "fsdp_7b_32gpu.yaml",
        "fsdp_13b_64gpu.yaml",
    ]
    for name in config_files:
        path = REPO_ROOT / "configs" / name
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg
        assert "training" in cfg
        assert "data" in cfg


def test_ddp_config_defaults_include_validation_and_logging():
    ddp_source = (REPO_ROOT / "train_ddp.py").read_text()
    assert "setdefault('val_split_ratio', 0.1)" in ddp_source
    assert "setdefault('json', False)" in ddp_source
    assert "setdefault('task_type', 'causal_lm')" in ddp_source

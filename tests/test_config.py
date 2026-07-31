"""Config defaults, merging and the CLI config builder."""

from vbmitigator.config import CFG


def test_defaults_have_core_sections():
    for section in ("EXPERIMENT", "MODEL", "SOLVER", "DATASET", "MITIGATOR", "OUTPUT"):
        assert section in CFG


def test_merge_from_file(tmp_path):
    yaml = tmp_path / "exp.yaml"
    yaml.write_text(
        "DATASET:\n  TYPE: 'utkface'\nMITIGATOR:\n  TYPE: 'badd'\nSOLVER:\n  LR: 0.05\n"
    )
    cfg = CFG.clone()
    cfg.merge_from_file(str(yaml))
    assert cfg.DATASET.TYPE == "utkface"
    assert cfg.MITIGATOR.TYPE == "badd"
    assert cfg.SOLVER.LR == 0.05


def test_cli_sets_config_stem(tmp_path):
    from vbmitigator.cli import _build_cfg

    yaml = tmp_path / "race.yaml"
    yaml.write_text("MITIGATOR:\n  TYPE: 'erm'\n")

    class Args:
        cfg = str(yaml)
        opts = []
        seed = 7
        epoch_steps = None

    cfg = _build_cfg(Args())
    assert cfg.EXPERIMENT.CONFIG == "race"
    assert cfg.EXPERIMENT.SEED == 7

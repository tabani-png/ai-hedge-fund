"""Seeding shipped mandates: fresh installs, upgrades, and deletions."""

from hedge_fund.paths import ensure_mandates_dir


def test_fresh_install_gets_every_shipped_mandate(tmp_path):
    d = ensure_mandates_dir(tmp_path / "mandates")
    assert {p.name for p in d.glob("*.yaml")} == {"example.yaml", "multi-repo.yaml"}


def test_upgrade_adds_the_new_mandate_but_not_a_deleted_example(tmp_path):
    d = tmp_path / "mandates"
    d.mkdir()
    (d / "mine.yaml").write_text("name: mine\n")  # an older install; its example was deleted
    ensure_mandates_dir(d)
    assert {p.name for p in d.glob("*.yaml")} == {"mine.yaml", "multi-repo.yaml"}


def test_deleted_mandate_stays_deleted(tmp_path):
    d = ensure_mandates_dir(tmp_path / "mandates")
    (d / "multi-repo.yaml").unlink()
    ensure_mandates_dir(d)
    assert not (d / "multi-repo.yaml").exists()


def test_user_edits_are_never_overwritten(tmp_path):
    d = tmp_path / "mandates"
    d.mkdir()
    (d / "multi-repo.yaml").write_text("name: my-edit\n")
    ensure_mandates_dir(d)
    assert (d / "multi-repo.yaml").read_text() == "name: my-edit\n"

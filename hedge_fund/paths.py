"""Where user data lives: ~/.hedge-fund/.

Everything the user owns — mandates, run/backtest receipts, API caches, and
the .env key file — lives under one home directory, outside the package. The
package directory stays read-only code, so a pipx install behaves exactly
like a checkout.

Textual-free and import-light on purpose: every layer (CLI, TUI, caches)
anchors its paths here, and nothing here may import them back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

USER_DIR = Path.home() / ".hedge-fund"
MANDATES_DIR = USER_DIR / "mandates"
CACHE_DIR = USER_DIR / "cache"
ENV_PATH = USER_DIR / ".env"

# The example mandate ships inside the package; it is copied out (never read
# in place) so users edit their copy, not the install.
EXAMPLE_MANDATE = Path(__file__).resolve().parent / "fund" / "example.yaml"
MULTI_REPO_MANDATE = Path(__file__).resolve().parent / "fund" / "multi-repo.yaml"


# Mandates that ship with the package, by the filename they are copied to.
SHIPPED_MANDATES = {"example.yaml": EXAMPLE_MANDATE, "multi-repo.yaml": MULTI_REPO_MANDATE}
_SEEDED_MARKER = ".seeded"


def ensure_mandates_dir(mandates_dir: Path | None = None) -> Path:
    """Create the mandates dir on first use, and seed each shipped mandate
    exactly once — including ones added in a later release, so upgrading
    users get them too. The `.seeded` marker remembers what was offered, so
    a mandate the user deleted is never resurrected."""
    mandates_dir = mandates_dir or MANDATES_DIR
    marker = mandates_dir / _SEEDED_MARKER
    if marker.exists():
        seeded = set(marker.read_text().split())
    elif mandates_dir.exists():
        seeded = {"example.yaml"}  # a pre-marker install already had its example
    else:
        seeded = set()
    mandates_dir.mkdir(parents=True, exist_ok=True)
    for name, source in SHIPPED_MANDATES.items():
        if name not in seeded:
            if not (mandates_dir / name).exists():
                shutil.copy(source, mandates_dir / name)
            seeded.add(name)
    marker.write_text("\n".join(sorted(seeded)) + "\n")
    return mandates_dir

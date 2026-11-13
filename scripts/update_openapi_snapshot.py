"""Regenerate docs/openapi_snapshot.json after an intentional schema change.

Review the diff: an API contract change needs a version bump and approval (CONTRIBUTING 8).
"""

from __future__ import annotations

import json

from driverisk.serving.mock_app import create_app
from driverisk.utils.paths import repo_root

SNAPSHOT = repo_root() / "docs" / "openapi_snapshot.json"


def main() -> None:
    text = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    SNAPSHOT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT}")


if __name__ == "__main__":
    main()

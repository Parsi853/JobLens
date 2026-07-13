"""Display persisted model metrics without inventing values."""

import json

from joblens.settings import load_settings


def main() -> None:
    """Print available JSON metric reports."""
    artifacts = load_settings().path("artifacts")
    found = False
    for path in sorted(artifacts.glob("*/metrics.json")):
        found = True
        values = json.loads(path.read_text(encoding="utf-8"))
        rendered = json.dumps(values, ensure_ascii=False, indent=2)
        print(f"\n{path.parent.name}:\n{rendered}")
    if not found:
        print("No trained model metrics found. Train models first.")


if __name__ == "__main__":
    main()

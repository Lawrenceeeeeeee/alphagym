"""Read-only ML research data preflight; writes only the requested audit artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphagym.reassessment_audit import audit_reassessment_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index-code", default="ALL_A")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_reassessment_data(args.root, index_code=args.index_code)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized if args.json else result["error"]["message"] if not result["ok"] else "OK")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

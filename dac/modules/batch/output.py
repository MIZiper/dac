import csv
import json
import os

from dac.core import DataNode, ActionNode
from dac.core.actions import ActionBase


class AppendCSVAction(ActionBase):
    CAPTION = "Append to CSV"

    def __call__(self, data: list[DataNode], filepath: str = "output.csv",
                 mode: str = "append") -> None:
        if not data:
            return

        file_exists = os.path.exists(filepath)

        rows = []
        all_keys = set()
        for node in data:
            row = _node_to_row(node)
            all_keys.update(row.keys())
            rows.append(row)

        fieldnames = sorted(all_keys)
        write_header = not file_exists or (file_exists and os.path.getsize(filepath) == 0)

        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


class AppendJSONLAction(ActionBase):
    CAPTION = "Append to JSONL"

    def __call__(self, data: list[DataNode], filepath: str = "output.jsonl") -> None:
        if not data:
            return

        with open(filepath, "a") as f:
            for node in data:
                f.write(json.dumps(_node_to_row(node), default=str) + "\n")


def _node_to_row(node: DataNode) -> dict:
    row = {"name": node.name}
    for k, v in node.__dict__.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool)):
            row[k] = v
        elif isinstance(v, (list, tuple)):
            row[k] = json.dumps(
                [_basic_val(e) for e in v], default=str
            )
        elif isinstance(v, dict):
            row[k] = json.dumps(
                {str(dk): _basic_val(dv) for dk, dv in v.items()}, default=str
            )
        else:
            row[k] = str(v)
    return row


def _basic_val(v):
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)

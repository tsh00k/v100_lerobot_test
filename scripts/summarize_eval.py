"""汇总 4_eval_libero.sh 的结果：读取 <OUT_ROOT>/<suite>/eval_info.json，打印 markdown 表格并写入 summary.md。

用法：python scripts/summarize_eval.py outputs/eval/<run>
"""

import json
import sys
from pathlib import Path

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main(out_root: Path) -> None:
    rows = []
    for info_file in sorted(out_root.glob("*/eval_info.json")):
        overall = json.loads(info_file.read_text())["overall"]
        rows.append((info_file.parent.name, overall["pc_success"], overall["n_episodes"], overall["eval_ep_s"]))
    if not rows:
        print(f"{out_root} 下没有 eval_info.json")
        return
    rows.sort(key=lambda r: SUITE_ORDER.index(r[0]) if r[0] in SUITE_ORDER else 99)

    lines = ["| suite | success (%) | episodes | s/episode |", "|---|---|---|---|"]
    lines += [f"| {s} | {p:.1f} | {n} | {t:.1f} |" for s, p, n, t in rows]
    if len(rows) > 1:
        total = sum(n for _, _, n, _ in rows)
        avg = sum(p for _, p, _, _ in rows) / len(rows)
        lines.append(f"| **average** | **{avg:.1f}** | {total} | |")
    table = "\n".join(lines)
    print(table)
    (out_root / "summary.md").write_text(table + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

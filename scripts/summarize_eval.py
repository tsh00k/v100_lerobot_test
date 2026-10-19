"""汇总 4_eval_libero.sh 的结果：读取 <OUT_ROOT>/<分片>/eval_info.json，按套件合并分片，
打印 markdown 表格并写入 summary.md。另附每个任务的成功率，便于定位失败集中在哪些任务。

用法：python scripts/summarize_eval.py outputs/eval/<run>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main(out_root: Path) -> None:
    succ = defaultdict(float)      # suite -> 成功 episode 数
    n_ep = defaultdict(int)        # suite -> episode 数
    secs = defaultdict(float)      # suite -> 各分片耗时之和（进程时间）
    per_task = defaultdict(dict)   # suite -> {task_id: pc_success}
    for info_file in sorted(out_root.glob("*/eval_info.json")):
        info = json.loads(info_file.read_text())
        suite = info_file.parent.name.split("_t")[0]
        o = info["overall"]
        succ[suite] += o["pc_success"] / 100 * o["n_episodes"]
        n_ep[suite] += o["n_episodes"]
        secs[suite] += o["eval_s"]
        for t in info.get("per_task", []):
            s = t["metrics"]["successes"]
            per_task[suite][t["task_id"]] = 100 * sum(s) / max(1, len(s))
    if not n_ep:
        print(f"{out_root} 下没有 eval_info.json")
        return

    suites = sorted(n_ep, key=lambda s: SUITE_ORDER.index(s) if s in SUITE_ORDER else 99)
    rate = {s: 100 * succ[s] / n_ep[s] for s in suites}
    lines = ["| suite | success (%) | episodes | s/episode |", "|---|---|---|---|"]
    lines += [f"| {s} | {rate[s]:.1f} | {n_ep[s]} | {secs[s] / n_ep[s]:.1f} |" for s in suites]
    if len(suites) > 1:
        lines.append(f"| **average** | **{sum(rate.values()) / len(suites):.1f}** | {sum(n_ep.values())} | |")
    lines += ["", "per-task success (%):", ""]
    for s in suites:
        tasks = per_task[s]
        lines.append(f"- {s}: " + ", ".join(f"{tid}:{tasks[tid]:.0f}" for tid in sorted(tasks)))
    table = "\n".join(lines)
    print(table)
    (out_root / "summary.md").write_text(table + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

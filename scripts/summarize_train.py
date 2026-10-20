"""汇总 3_train_lora.sh 一次训练的耗时和资源占用，打印 markdown 并写入 <MON_DIR>/summary.md。

输入（都由 3_train_lora.sh 写在 <MON_DIR> 下）：
  train.log  lerobot_train 的输出。每个日志窗口一行，含 updt_s / data_s / mem_gb
             （lerobot 0.6.1 中三者都取所有 rank 的最大值，见 lerobot_train.py 的 train_metrics）
  gpu.csv    nvidia-smi 采样（仅 MONITOR=full）：time,index,memory.used,utilization.gpu
  host.csv   主机采样（MONITOR=full / cpu）：time,used_mib,loadavg_1m,<cpu jiffies...>
  wall_s     整次运行的墙钟时间（秒，含模型加载）

用法：python scripts/summarize_train.py <MON_DIR> [目标步数，用于估算正式训练耗时]
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_RE = re.compile(r"step:(\S+) .*updt_s:([\d.]+) data_s:([\d.]+)(?:.*mem_gb:([\d.]+))?")


def main(mon: Path, target_steps: int) -> None:
    windows = [m.groups() for m in map(LOG_RE.search, (mon / "train.log").read_text(errors="ignore").splitlines()) if m]
    lines = []
    if windows:
        # 第一个窗口包含 cudnn 预热、dataloader 启动，单独列出，不计入稳态
        steady = windows[1:] or windows
        updt = sum(float(w[1]) for w in steady) / len(steady)
        data = sum(float(w[2]) for w in steady) / len(steady)
        step_s = updt + data
        mem = max((float(w[3]) for w in windows if w[3]), default=float("nan"))
        lines += [
            # 输出用英文：tmux / 终端 locale 不是 UTF-8 时中文会显示成 "_"
            "| metric | value |", "|---|---|",
            f"| steady s/step (updt + data, slowest rank) | **{step_s:.3f}** |",
            f"| updt_s / data_s | {updt:.3f} / {data:.3f} |",
            f"| first window s/step (warm-up) | {float(windows[0][1]) + float(windows[0][2]):.3f} |",
            f"| log windows (steady / total) | {len(steady)} / {len(windows)} |",
            f"| torch peak mem, max_memory_allocated (GB) | {mem:.2f} |",
            f"| est. {target_steps} steps (h, excl. load/save) | {step_s * target_steps / 3600:.1f} |",
        ]
    else:
        lines += ["no updt_s / data_s lines in train.log (LOG_FREQ > total steps?)", ""]

    wall = mon / "wall_s"
    if wall.exists():
        lines.append(f"| wall time of whole run (min) | {int(wall.read_text()) / 60:.1f} |")

    gpu = mon / "gpu.csv"
    if gpu.exists():
        mem_peak, util = defaultdict(int), defaultdict(list)
        for row in csv.reader(gpu.open()):
            if len(row) < 4 or not row[1].strip().isdigit():
                continue
            i, m, u = int(row[1]), int(row[2]), int(row[3])
            mem_peak[i] = max(mem_peak[i], m)
            if m > 2000:  # 只统计模型已加载、在训练中的采样点
                util[i].append(u)
        lines += ["", "nvidia-smi (incl. CUDA context + allocator cache, larger than torch's number):", "",
                  "| GPU | peak mem (GiB) | mean util (%) |", "|---|---|---|"]
        for i in sorted(mem_peak):
            avg = sum(util[i]) / len(util[i]) if util[i] else 0
            lines.append(f"| {i} | {mem_peak[i] / 1024:.1f} | {avg:.0f} |")

    host = mon / "host.csv"
    if host.exists():
        rows = [r for r in csv.reader(host.open()) if len(r) >= 7 and r[1].isdigit()]
        if rows:
            # /proc/stat cpu 行：user nice system idle iowait irq softirq ...；idle + iowait 算空闲
            jif = [[int(x) for x in r[3:]] for r in rows]
            busy = []
            for a, b in zip(jif, jif[1:]):
                total = sum(b) - sum(a)
                idle = (b[3] + b[4]) - (a[3] + a[4])
                if total > 0:
                    busy.append(100 * (total - idle) / total)
            load = [float(r[2]) for r in rows]
            lines += ["", "host (whole machine, incl. other users):", "",
                      "| metric | mean | peak |", "|---|---|---|",
                      f"| CPU util (%, all cores) | {sum(busy) / len(busy):.0f} | {max(busy):.0f} |" if busy
                      else "| CPU util (%) | too few samples | |",
                      f"| load avg 1m ({os.cpu_count()} cores) | {sum(load) / len(load):.1f} | {max(load):.1f} |",
                      f"| used RAM (GiB) | {sum(int(r[1]) for r in rows) / len(rows) / 1024:.1f} | {max(int(r[1]) for r in rows) / 1024:.1f} |"]

    text = "\n".join(lines)
    print(text)
    (mon / "summary.md").write_text(text + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30000)

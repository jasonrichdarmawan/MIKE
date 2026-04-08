import os, re, subprocess
from datetime import datetime, timedelta

BASE    = "/lambda/nfs/Llama-3-8B-Analysis/CaKE"
_log_candidates = [f"{BASE}/logs/run_analysis.log", f"{BASE}/logs/run_eap_alphaedit.log"]
LOG = max(
    (p for p in _log_candidates if os.path.exists(p)),
    key=os.path.getmtime,
    default=f"{BASE}/logs/run_analysis.log",
)
DATA    = f"{BASE}/datasets/meta-llama/Meta-Llama-3-8B"
AE_OUT  = f"{DATA}/composition_incorrect_alphaedit"
IFM_OUT = f"{DATA}/composition_incorrect_ifmet"
EAP_OUT = f"{DATA}/composition_incorrect_eap_alphaedit"

STEPS = [
    "AlphaEdit edit + analysis",
    "analyze_tables: AlphaEdit",
    "IFMET edit + analysis",
    "analyze_tables: IFMET",
    "EAP-AlphaEdit edit + analysis",
    "analyze_tables: EAP-AlphaEdit",
]

# Estimated durations (seconds)
# Llama-3-8B on A100 40GB, batch=128
EST = {
    "AlphaEdit edit + analysis":       60 * 60 * 8,  # ~8h measured (batch=128)
    "analyze_tables: AlphaEdit":       5  * 60,
    "IFMET edit + analysis":           60 * 60 * 8,  # ~8h measured (batch=128)
    "analyze_tables: IFMET":           5  * 60,
    "EAP-AlphaEdit edit + analysis":   60 * 60 * 8,  # ~8h measured (batch=128)
    "analyze_tables: EAP-AlphaEdit":   5  * 60,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_hm(secs):
    h, rem = divmod(max(0, int(secs)), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"

def parse_hms(s):
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0]) * 60 + int(parts[1])

def parse_tqdm(text):
    pat = re.compile(r"\d+%\|.*?\|\s*(\d+)/(\d+)\s*\[[\d:]+<([\d:]+),\s*([\d.]+)s/")
    m = pat.findall(text)
    if not m:
        return None
    done, total, eta_str, rate_str = m[-1]
    return int(done), int(total), parse_hms(eta_str), float(rate_str)

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError):
        return False

def find_pipeline_pid():
    for pattern in ["run_analysis.sh", "run_analysis.py"]:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        pids = r.stdout.strip().split()
        if pids:
            return int(pids[0])
    return None

# ── Read log ───────────────────────────────────────────────────────────────────

now = datetime.now()

if os.path.exists(LOG) and os.path.getmtime(LOG) > (now.timestamp() - 86400):
    log = open(LOG, "rb").read().decode("utf-8", errors="replace")
else:
    log = ""

# ── Step status ────────────────────────────────────────────────────────────────

step_status = {}
for step in STEPS:
    if f"{step} DONE" in log:
        step_status[step] = "done"
    elif f"{step} FAILED" in log:
        step_status[step] = "failed"
    elif f"] {step} ==========" in log and f"{step} DONE" not in log and f"{step} FAILED" not in log:
        step_status[step] = "running"
    else:
        step_status[step] = "pending"

def log_section(step):
    marker = f"] {step} =========="
    idx = log.rfind(marker)
    return log[idx:] if idx >= 0 else ""

# ── Batch progress for edit+analysis steps ────────────────────────────────────

def batch_progress(step):
    """Return (batches_done, batches_total, items_done, items_total, current_phase) from log, or None."""
    section = log_section(step)
    # match "--- AlphaEdit Batch X/N", "--- IFMET Batch X/N", or "--- EAP-AlphaEdit Batch X/N"
    batch_pat = re.findall(r"Batch\s+(\d+)/(\d+)", section)
    item_pat  = re.findall(r"Item\s+(\d+)/(\d+)\s+edited", section)
    if not batch_pat:
        return None
    b_cur, b_total = int(batch_pat[-1][0]), int(batch_pat[-1][1])
    # count completed batches (batches where "Weights restored" appears after them)
    restored = section.count("Weights restored to original")
    i_cur  = int(item_pat[-1][0])  if item_pat  else 0
    i_size = int(item_pat[-1][1])  if item_pat  else 0
    # detect current phase from last [Phase: X] marker
    phase_pat = re.findall(r"\[Phase:\s*([^\]]+)\]", section)
    # if still editing items within current batch (no phase marker yet for this batch)
    if phase_pat:
        last_phase = phase_pat[-1].strip()
        # check if weights were restored after this phase marker (batch finished)
        last_phase_idx = section.rfind(f"[Phase: {last_phase}]")
        restored_after = section.count("Weights restored to original",
                                       last_phase_idx + len(f"[Phase: {last_phase}]"))
        current_phase = "Edit" if restored_after > 0 else last_phase
    else:
        current_phase = "Edit"
    return restored, b_total, i_cur, i_size, current_phase

# ── Remaining time estimates ───────────────────────────────────────────────────

def step_start_time(step):
    """Parse step start timestamp from log, return datetime or None."""
    marker = f"] {step} =========="
    for line in log.splitlines():
        if marker in line:
            m = re.search(r"\[(\d+:\d+:\d+)\]", line)
            if m:
                t = datetime.strptime(m.group(1), "%H:%M:%S").replace(
                    year=now.year, month=now.month, day=now.day)
                if t > now:  # rolled past midnight
                    t -= timedelta(days=1)
                return t
    return None

def measured_secs_per_batch(step):
    """Return actual seconds/batch from log timing, or None."""
    bp = batch_progress(step)
    if not bp:
        return None
    done_batches = bp[0]
    if done_batches == 0:
        return None
    start = step_start_time(step)
    if not start:
        return None
    return (now - start).total_seconds() / done_batches

def step_remaining(step):
    s = step_status[step]
    if s in ("done", "failed"):
        return 0
    if "edit + analysis" in step:
        bp = batch_progress(step)
        if bp:
            done_batches, total_batches, _, _, _ = bp
            if total_batches > 0 and done_batches > 0:
                spb = measured_secs_per_batch(step)
                if spb:
                    return spb * (total_batches - done_batches)
        # pending — try to borrow rate from another completed edit+analysis step
        edit_steps = [s for s in STEPS if "edit + analysis" in s and s != step]
        for other in edit_steps:
            spb = measured_secs_per_batch(other)
            if spb:
                bp_other = batch_progress(other)
                n_batches = bp_other[1] if bp_other else 65
                return spb * n_batches
    return EST[step]

# ── Totals ─────────────────────────────────────────────────────────────────────

remaining_per_step = {step: step_remaining(step) for step in STEPS}
total_secs = sum(remaining_per_step[s] for s in STEPS if step_status[s] not in ("done", "failed"))
eta_finish = now + timedelta(seconds=total_secs)

# ── Print ──────────────────────────────────────────────────────────────────────

print(f"\n  Pipeline monitor  [{now.strftime('%H:%M:%S')}]")
print(f"  {'─'*55}")
print(f"\n  Pipeline  [~{fmt_hm(total_secs)} remaining,  ETA {eta_finish.strftime('%a %H:%M')}]")

cursor = now
for step in STEPS:
    s   = step_status[step]
    rem = remaining_per_step[step]
    eta_step = cursor + timedelta(seconds=rem)

    if s == "done":
        ts = ""
        for line in log.splitlines():
            if f"{step} DONE" in line:
                ts = re.search(r"\[(\d+:\d+:\d+)\]", line)
                ts = ts.group(1) if ts else ""
                break
        print(f"    ✓  {step}  ({ts})")

    elif s == "failed":
        print(f"    ✗  {step}  FAILED")

    elif s == "running":
        print(f"    ▶  {step}")
        if "edit + analysis" in step:
            bp = batch_progress(step)
            if bp:
                done_batches, total_batches, i_cur, i_size, current_phase = bp
                active_batch = done_batches + 1 if done_batches < total_batches else total_batches
                bar_w  = 22
                filled = int(bar_w * done_batches / total_batches) if total_batches else 0
                bar    = "█" * filled + "░" * (bar_w - filled)
                pct    = int(100 * done_batches / total_batches) if total_batches else 0
                print(f"       [{bar}] {done_batches}/{total_batches} batches done ({pct}%)")
                phase_label = current_phase if current_phase != "Edit" or not i_size else f"Edit  item {i_cur}/{i_size}"
                print(f"       Batch {active_batch}/{total_batches}  phase: {phase_label}")
            # tqdm for current inference pass
            section = log_section(step)
            t = parse_tqdm(section)
            if t:
                done_ex, total_ex, eta_cur, rate = t
                pct2 = int(100 * done_ex / total_ex)
                b2   = "█" * int(18 * done_ex / total_ex) + "░" * (18 - int(18 * done_ex / total_ex))
                print(f"       [{b2}] {pct2}%  ~{fmt_hm(eta_cur)} left (inference)")
        print(f"       ~{fmt_hm(rem)}  →  {eta_step.strftime('%a %H:%M')}")
        cursor += timedelta(seconds=rem)

    else:  # pending
        print(f"    ·  {step}  ~{fmt_hm(rem)}  starts {cursor.strftime('%a %H:%M')}")
        cursor += timedelta(seconds=rem)

# ── GPU ────────────────────────────────────────────────────────────────────────
print()
r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True)
parts = [p.strip() for p in r.stdout.strip().split(",")]
if len(parts) == 3:
    util = parts[0]
    mem_used = int(parts[1].replace(" MiB", ""))
    mem_total = int(parts[2].replace(" MiB", ""))
    mem_pct = mem_used / mem_total
    bar_w = 20
    filled = int(bar_w * mem_pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    print(f"  GPU util  {util}")
    print(f"  VRAM     [{bar}] {mem_used}/{mem_total} MiB ({mem_pct*100:.0f}%)")
else:
    print(f"  GPU      {r.stdout.strip()}")

# ── Process ────────────────────────────────────────────────────────────────────
pid = find_pipeline_pid()
if pid and pid_alive(pid):
    r2 = subprocess.run(["ps", "-o", "pid,etime,pcpu", "--no-headers", "-p", str(pid)],
                        capture_output=True, text=True)
    print(f"  Process  {r2.stdout.strip()}")
else:
    print(f"  Process  *** not running ***")

# ── Last log line ──────────────────────────────────────────────────────────────
if os.path.exists(LOG):
    lines = [l for l in log.splitlines() if l.strip() and "====" not in l and "█" not in l and "░" not in l]
    if lines:
        print(f"  Last     {lines[-1].strip()[:80]}")
print()

"""Resumable driver for the experiment manifest.

Designed to be left alone for days: it appends one CSV row per completed run,
skips anything already recorded on restart, isolates per-run failures, and
writes a heartbeat so progress can be checked without attaching to the process.
Combined with a systemd unit using Restart=always, a crash or reboot costs at
most the run that was in flight.
"""
import argparse
import csv
import glob
import json
import logging
import os
import shutil
import signal
import socket
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnti import tracking  # noqa: E402
from scripts.experiment import run_one  # noqa: E402

logger = logging.getLogger("driver")

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    logger.warning("received signal %s; finishing current run then stopping", signum)
    _STOP = True


def load_manifest(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def completed_ids(results_dir):
    """Union of run ids finished by *any* host.

    Each host owns its own CSV (concurrent appends to a single file across NFS
    would interleave and corrupt rows), but every host reads all of them, so
    re-sharding or re-running never repeats completed work.
    """
    done = set()
    for path in glob.glob(os.path.join(results_dir, "*.csv")):
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("id"):
                        done.add(row["id"])
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
    return done


def in_shard(spec, index, total):
    """Hash-partition the manifest. Run ids are already sha1 hex digests, so
    they distribute evenly without any coordination between hosts."""
    if total <= 1:
        return True
    return int(spec["id"], 16) % total == index


def resolve_shard(hosts_file, hostname, fallback_index, fallback_total):
    """Derive this host's shard from its own name and the shared hosts file.

    The shard index deliberately is NOT baked into the systemd unit: ~/.config
    lives on NFS-shared /home, so every host writes and reads the *same* unit
    file, and a per-host substituted value is silently overwritten by whichever
    host was configured last. That made all 13 machines run an identical shard.
    Deriving it at runtime from the hostname removes the failure mode entirely.
    """
    if not hosts_file:
        return fallback_index, fallback_total
    try:
        with open(hosts_file) as fh:
            hosts = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    except OSError as exc:
        logger.warning("cannot read %s (%s); falling back to explicit shard", hosts_file, exc)
        return fallback_index, fallback_total

    if hostname not in hosts:
        raise SystemExit(
            f"host {hostname!r} is not listed in {hosts_file}; refusing to guess a shard "
            f"(would duplicate another host's work). Known hosts: {hosts}"
        )
    return hosts.index(hostname), len(hosts)


def host_fingerprint():
    """Recorded per run so a machine effect can be tested for, not assumed away."""
    name = socket.gethostname().split(".")[0]
    gpu = "unknown"
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return name, gpu


def append_row(results_path, row, fieldnames_cache={}):
    """Append a row, rewriting the header if new columns appear."""
    exists = os.path.exists(results_path)
    existing = []
    if exists:
        with open(results_path, newline="") as fh:
            reader = csv.DictReader(fh)
            existing = list(reader)
            header = reader.fieldnames or []
    else:
        header = []

    new_cols = [k for k in row if k not in header]
    if new_cols and exists:
        header = header + new_cols
        with open(results_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for r in existing:
                w.writerow(r)
            w.writerow(row)
        return
    if not exists:
        header = list(row)
        with open(results_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=header).writeheader()
    with open(results_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header).writerow(row)


def free_gb(path):
    return shutil.disk_usage(path).free / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifest.jsonl")
    ap.add_argument("--results-dir", default="experiments/results",
                    help="per-host CSVs live here; every host reads all of them")
    ap.add_argument("--artifacts", default="/scratch/mdra00001/nnti-artifacts")
    ap.add_argument("--heartbeat-dir", default="experiments/heartbeat")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-from-hosts", default=None,
                    help="derive shard index/total from this host's position in the "
                         "hosts file; overrides --shard-index/--num-shards")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N runs")
    ap.add_argument("--stages", default=None)
    ap.add_argument("--wandb-project", default="nnti-lipophilicity-rerun")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default=None,
                    help="online | offline | off (default: online if credentials exist)")
    args = ap.parse_args()

    wandb_mode = args.wandb_mode or tracking.default_mode()
    hostname, gpu_name = host_fingerprint()
    shard_index, num_shards = resolve_shard(
        args.shard_from_hosts, hostname, args.shard_index, args.num_shards)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.heartbeat_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{hostname}.csv")
    heartbeat_path = os.path.join(args.heartbeat_dir, f"{hostname}.json")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    os.makedirs(args.artifacts, exist_ok=True)

    manifest = load_manifest(args.manifest)
    if args.stages:
        keep = set(args.stages.split(","))
        manifest = [s for s in manifest if s.get("stage") in keep]

    mine = [s for s in manifest if in_shard(s, shard_index, num_shards)]
    done = completed_ids(args.results_dir)
    pending = [s for s in mine if s["id"] not in done]
    logger.info("host=%s shard=%d/%d manifest=%d mine=%d done_anywhere=%d pending=%d",
                hostname, shard_index, num_shards,
                len(manifest), len(mine), len(done), len(pending))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s gpu=%s", device, gpu_name)

    started = time.time()
    n_ok = n_fail = 0

    for idx, spec in enumerate(pending):
        if _STOP:
            logger.info("stopping cleanly at user request")
            break
        if args.limit and idx >= args.limit:
            logger.info("reached --limit=%d", args.limit)
            break
        if free_gb(args.artifacts) < args.min_free_gb:
            logger.error("only %.1f GB free on artifacts volume; pausing",
                         free_gb(args.artifacts))
            break

        t0 = time.time()
        row = {"id": spec["id"], "stage": spec.get("stage"), "kind": spec.get("kind"),
               "host": hostname, "gpu": gpu_name}
        row.update({k: v for k, v in spec.items() if k not in ("id", "stage", "kind")})

        tracker = tracking.start_run(spec, args.wandb_project, args.wandb_entity, wandb_mode)
        try:
            logger.info("[%d/%d] %s %s", idx + 1, len(pending), spec["stage"], spec["id"])
            result = run_one(spec, args.artifacts, device=device)
            history = result.pop("_history", [])
            row.update(result)
            row["status"] = "ok"
            tracker.log_history(history)
            tracker.log_summary(
                {k: v for k, v in result.items() if isinstance(v, (int, float, bool, str))}
            )
            tracker.finish("ok")
            n_ok += 1
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            row.update({"status": "oom", "error": "CUDA OOM"})
            logger.error("run %s hit CUDA OOM", spec["id"])
            tracker.finish("oom")
            n_fail += 1
        except Exception as exc:  # keep the sweep alive
            torch.cuda.empty_cache()
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]})
            logger.error("run %s failed:\n%s", spec["id"], traceback.format_exc())
            tracker.finish("error")
            n_fail += 1

        row.pop("_history", None)  # never let a nested list reach the CSV
        row["wall_seconds"] = round(time.time() - t0, 2)
        append_row(results_path, row)

        elapsed = time.time() - started
        rate = (idx + 1) / elapsed if elapsed else 0
        with open(heartbeat_path, "w") as fh:
            json.dump({
                "host": hostname,
                "shard": f"{shard_index}/{num_shards}",
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_this_session": idx + 1,
                "pending_total": len(pending),
                "ok": n_ok, "failed": n_fail,
                "runs_per_hour": round(rate * 3600, 2),
                "eta_hours": round((len(pending) - idx - 1) / rate / 3600, 2) if rate else None,
                "last_run": spec["id"], "last_stage": spec.get("stage"),
            }, fh, indent=2)

    logger.info("session done: ok=%d failed=%d in %.1f h", n_ok, n_fail,
                (time.time() - started) / 3600)


if __name__ == "__main__":
    main()

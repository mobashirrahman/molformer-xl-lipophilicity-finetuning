"""Weights & Biases logging for the sweep.

Every method here is failure-tolerant by design. This runs unattended for days,
so a network blip, an expired credential, or a W&B outage must degrade to
"no telemetry for this run" and never abort the experiment that is already
mid-flight. All W&B calls are wrapped; the driver's CSV remains the source of
truth and is written whether or not tracking succeeds.
"""
import logging
import os

logger = logging.getLogger(__name__)

# Guard rail: a hung wandb.init would otherwise stall the sweep indefinitely.
INIT_TIMEOUT_SECONDS = 60


class NullTracker:
    """Stand-in used when tracking is disabled or has failed."""

    active = False

    def log_history(self, history):
        pass

    def log_summary(self, metrics):
        pass

    def finish(self, status="ok"):
        pass


class WandbTracker:
    def __init__(self, run):
        self._run = run
        self.active = True

    def log_history(self, history):
        """Log the per-epoch curve; step is the epoch number."""
        if not history:
            return
        try:
            import wandb  # noqa: F401
            for entry in history:
                self._run.log({k: v for k, v in entry.items() if k != "epoch"},
                              step=entry.get("epoch"))
        except Exception as exc:
            logger.warning("wandb history logging failed: %s", exc)

    def log_summary(self, metrics):
        try:
            for key, value in metrics.items():
                self._run.summary[key] = value
        except Exception as exc:
            logger.warning("wandb summary logging failed: %s", exc)

    def finish(self, status="ok"):
        try:
            self._run.summary["run_status"] = status
            self._run.finish(exit_code=0 if status == "ok" else 1)
        except Exception as exc:
            logger.warning("wandb finish failed: %s", exc)


def start_run(spec, project, entity=None, mode="online", extra_tags=()):
    """Begin a tracked run, or return a NullTracker if tracking is unavailable."""
    if mode in ("off", "disabled", None):
        return NullTracker()

    try:
        import wandb

        stage = spec.get("stage", "unknown")
        tags = [f"stage:{stage}", f"kind:{spec.get('kind', 'unknown')}"]
        for key in ("strategy", "method", "influence_scope"):
            if spec.get(key):
                tags.append(f"{key}:{spec[key]}")
        tags.extend(extra_tags)

        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            group=f"stage-{stage}",
            job_type=spec.get("kind"),
            name=f"{stage}-{spec['id']}",
            id=spec["id"],
            config=dict(spec),
            tags=tags,
            reinit="finish_previous",
            resume="allow",
            settings=wandb.Settings(init_timeout=INIT_TIMEOUT_SECONDS, silent=True),
        )
        return WandbTracker(run)
    except Exception as exc:
        # Never let telemetry take down the sweep.
        logger.warning("wandb init failed for %s (%s); continuing untracked",
                       spec.get("id"), exc)
        return NullTracker()


def default_mode():
    """Offline unless credentials are actually present."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    netrc = os.environ.get("NETRC", os.path.expanduser("~/.netrc"))
    if os.environ.get("WANDB_API_KEY") or os.path.exists(netrc):
        return "online"
    return "offline"

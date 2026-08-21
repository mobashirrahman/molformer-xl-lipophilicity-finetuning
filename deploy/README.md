# Unattended execution

The sweep runs as a systemd **user** service so it survives logout and reboot.

```bash
cp deploy/nnti-experiments.service ~/.config/systemd/user/
loginctl enable-linger "$USER"          # required to survive logout/reboot
systemctl --user daemon-reload
systemctl --user enable --now nnti-experiments.service
```

Monitoring:

```bash
systemctl --user status nnti-experiments.service
cat experiments/heartbeat.json           # progress, throughput, ETA
tail -f /scratch/mdra00001/nnti-experiments.log
```

The driver is resumable: it reads `experiments/results.csv` on start and skips
run ids already recorded, so `Restart=always` makes a crash or reboot cost at
most the single run that was in flight. Stopping is safe at any point —
`systemctl --user stop` lets the current run finish first.

Paths point at `/scratch` deliberately. The home volume has a per-user quota
that the HF model cache, pip cache, and W&B state will exhaust; `pip`'s cache
alone had consumed ~8 GB before this work started.

## Distributed execution across the lab machines

`/home` is NFS-shared, so code, manifest and results are visible on every
machine with no copying. `/scratch` is local disk per host, so the venv, HF
cache and checkpoints have to be replicated.

```bash
./scripts/cluster.sh discover        # writes experiments/hosts.txt
./scripts/cluster.sh bootstrap       # rsync venv + caches + credentials
STAGES=B ./scripts/cluster.sh start  # run stage B everywhere
./scripts/cluster.sh status
./scripts/cluster.sh sync-artifacts  # after B: push checkpoints to all hosts
STAGES=C1,C2,C3,D ./scripts/cluster.sh start
./scripts/cluster.sh stop
```

**Sharding** is a hash partition of the run id (`int(id,16) % num_shards`).
Run ids are sha1 digests, so the split is even and needs no coordination
between hosts — nothing to lock, nothing to claim.

**Results**: each host appends to `experiments/results/<host>.csv`. Concurrent
appends to one file over NFS would interleave and corrupt rows. Every host
reads *all* the CSVs when building its skip-list, so re-sharding or restarting
never repeats completed work, and hosts can be added or removed mid-flight.

**The one ordering constraint**: stage B writes `baseline_split*.pt`, which
C2/C3/D load. Those land on whichever host ran them, so `sync-artifacts` must
run between B and the later stages.

The venv is rsynced rather than rebuilt per host, deliberately: identical
absolute paths mean it relocates cleanly, and it guarantees bit-identical
package versions. A per-host `pip install` could drift and would confound
results this study is specifically trying to resolve near the noise floor.

Every result row records `host` and `gpu` so a machine effect can be tested
for rather than assumed away. All lab machines are RTX 2060 SUPER, so the
hardware is homogeneous, but the columns make that checkable.

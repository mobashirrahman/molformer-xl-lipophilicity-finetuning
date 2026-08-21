#!/usr/bin/env bash
# Control the sweep across the bio* lab machines.
#
#   ./scripts/cluster.sh discover           # find reachable hosts with a working GPU
#   ./scripts/cluster.sh bootstrap          # replicate env to every host in HOSTS_FILE
#   ./scripts/cluster.sh sync-artifacts     # push baseline checkpoints everywhere
#   ./scripts/cluster.sh collect            # gather per-config influence scores
#   ./scripts/cluster.sh start              # install + start the shard on every host
#   ./scripts/cluster.sh hpo [trials]       # baseline HPO worker on every host
#   ./scripts/cluster.sh hpo-peft [trials]  # per-method PEFT HPO on every host
#   ./scripts/cluster.sh hpo-lora [trials]  # re-tune LoRA alone (widened range)
#   ./scripts/cluster.sh hpo-stop           # stop both HPO services
#   ./scripts/cluster.sh status
#   ./scripts/cluster.sh stop
#
# /home is NFS-shared, so code, manifest and results are visible everywhere with
# no copying. /scratch is local disk per machine, so the venv, HF cache and
# checkpoints must be replicated to each host.
set -uo pipefail

USER_NAME="$(id -un)"
PROJECT="/home/${USER_NAME}/NNTI_Project"
SCRATCH="/scratch/${USER_NAME}"
VENV="${SCRATCH}/nnti-venv"
HOSTS_FILE="${PROJECT}/experiments/hosts.txt"
DOMAIN="studcs.uni-saarland.de"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

hosts() { grep -vE '^\s*(#|$)' "$HOSTS_FILE"; }

discover() {
  echo "probing bio00..bio24 ..."
  : > "${HOSTS_FILE}.tmp"
  for n in $(seq -w 0 24); do
    (
      h="bio${n}.${DOMAIN}"
      free=$(timeout 25 ssh $SSH_OPTS "$h" \
        'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1' 2>/dev/null)
      # Require a working GPU with most of its memory available.
      if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -gt 6000 ]; then
        echo "bio${n}" >> "${HOSTS_FILE}.tmp"
      fi
    ) &
  done
  wait
  sort -u "${HOSTS_FILE}.tmp" > "$HOSTS_FILE" && rm -f "${HOSTS_FILE}.tmp"
  echo "usable hosts ($(hosts | wc -l)):"; hosts | tr '\n' ' '; echo
}

bootstrap() {
  local n; n=$(hosts | wc -l); local i=0
  for h in $(hosts); do
    (
      fq="${h}.${DOMAIN}"
      # Never rsync --delete a path onto itself.
      if [ "$h" = "$(hostname -s)" ]; then echo "  ${h}: local (source host, skipped)"; exit 0; fi
      # rsync the venv rather than rebuilding: identical paths everywhere means
      # it relocates cleanly, and it guarantees bit-identical package versions.
      # Rebuilding per host could silently drift and confound the results.
      ssh $SSH_OPTS "$fq" "mkdir -p ${SCRATCH}/nnti-artifacts" 2>/dev/null
      rsync -a --delete -e "ssh $SSH_OPTS" "${VENV}/"        "${fq}:${VENV}/"        2>/dev/null
      rsync -a           -e "ssh $SSH_OPTS" "${SCRATCH}/hf-cache/"    "${fq}:${SCRATCH}/hf-cache/"    2>/dev/null
      rsync -a           -e "ssh $SSH_OPTS" "${SCRATCH}/wandb-netrc"  "${fq}:${SCRATCH}/wandb-netrc"  2>/dev/null
      ok=$(ssh $SSH_OPTS "$fq" "${VENV}/bin/python -c 'import torch;print(torch.cuda.is_available())'" 2>/dev/null)
      echo "  ${h}: cuda=${ok:-FAILED}"
    ) &
    i=$((i+1)); [ $((i % 6)) -eq 0 ] && wait
  done
  wait
  echo "bootstrap done ($n hosts)"
}

sync_artifacts() {
  # Stage B writes baseline_*.pt on whichever host ran it; every host
  # needs all of them before C2/C3/D can start. The glob is deliberately
  # baseline_*.pt rather than baseline_split*.pt, because stage F1 writes
  # model-namespaced checkpoints (baseline_ChemBERTa-zinc-base-v1_split*.pt)
  # that F2 loads, and the narrower pattern would silently skip them.
  #
  # These are relayed host-to-host through one hub's /scratch, never staged on
  # /home. An earlier version collected them into the repo and exhausted the
  # NFS home quota at ~1 GB, which stopped every write on the machine. ~/.ssh
  # is itself NFS-shared, so the hub can reach every other host directly.
  local hub; hub=$(hosts | head -1)
  local me;  me=$(hostname -s)
  local dir="${SCRATCH}/nnti-artifacts"
  echo "hub=${hub}; gathering checkpoints from all hosts ..."

  # This host may hold checkpoints even when it is absent from HOSTS_FILE (for
  # example after discover() dropped it for a busy GPU), so include it as a
  # source unconditionally.
  local sources; sources=$(printf '%s\n' $(hosts) "$me" | sort -u)

  for h in $sources; do
    [ "$h" = "$hub" ] && continue
    ssh $SSH_OPTS "${hub}.${DOMAIN}" \
      "rsync -a --ignore-existing -e 'ssh ${SSH_OPTS}' \
         '${h}.${DOMAIN}:${dir}/baseline_*.pt' '${dir}/'" 2>/dev/null
  done

  local n; n=$(ssh $SSH_OPTS "${hub}.${DOMAIN}" "ls -1 ${dir}/baseline_*.pt 2>/dev/null | wc -l")
  echo "hub holds ${n:-0} checkpoints; pushing to every host ..."
  for h in $(hosts); do
    [ "$h" = "$hub" ] && continue
    ( ssh $SSH_OPTS "${hub}.${DOMAIN}" \
        "rsync -a --ignore-existing -e 'ssh ${SSH_OPTS}' \
           ${dir}/baseline_*.pt '${h}.${DOMAIN}:${dir}/'" 2>/dev/null
      # Verify by counting what actually landed, never by trusting rsync's exit
      # status through two levels of ssh. A push once reported success while the
      # host received nothing; the next stage would have failed on a missing
      # checkpoint after already spending the GPU time that produced it.
      got=$(ssh $SSH_OPTS "${h}.${DOMAIN}" "ls -1 ${dir}/baseline_*.pt 2>/dev/null | wc -l")
      if [ "${got:-0}" -eq "${n:-0}" ]; then
        echo "  ${h}: ${got}/${n} ok"
      else
        echo "  ${h}: ${got:-0}/${n} MISMATCH - retrying"
        ssh $SSH_OPTS "${hub}.${DOMAIN}" \
          "rsync -a --ignore-existing -e 'ssh ${SSH_OPTS}' \
             ${dir}/baseline_*.pt '${h}.${DOMAIN}:${dir}/'"
        got=$(ssh $SSH_OPTS "${h}.${DOMAIN}" "ls -1 ${dir}/baseline_*.pt 2>/dev/null | wc -l")
        [ "${got:-0}" -eq "${n:-0}" ] && echo "  ${h}: ${got}/${n} ok after retry" \
                                     || echo "  ${h}: ${got:-0}/${n} STILL FAILING"
      fi ) &
  done
  wait
}

collect() {
  # Stage C1 writes per-config influence scores to local /scratch on whichever
  # host ran them. Analysis needs them all in one place; they are small (~20 KB
  # each), so unlike the checkpoints these are safe to gather onto /home.
  local dest="${PROJECT}/experiments/influence"
  mkdir -p "$dest"
  for h in $(hosts); do
    ( rsync -a --ignore-existing -e "ssh $SSH_OPTS" \
        "${h}.${DOMAIN}:${SCRATCH}/nnti-artifacts/influence/" "$dest/" 2>/dev/null
      echo "  ${h}: collected" ) &
  done
  wait
  echo "influence score files: $(ls -1 "$dest" 2>/dev/null | wc -l)"
}

start() {
  local total; total=$(hosts | wc -l); local idx=0
  for h in $(hosts); do
    (
      fq="${h}.${DOMAIN}"
      ssh $SSH_OPTS "$fq" "mkdir -p ~/.config/systemd/user" 2>/dev/null
      # Only STAGES is substituted, and it is identical on every host. The
      # shard index must NOT be baked in: ~/.config is on NFS-shared /home, so
      # all hosts share one unit file and a per-host value would be clobbered.
      sed -e "s|__STAGES__|${STAGES:-B,C1,C2}|" \
          "${PROJECT}/deploy/nnti-shard.service.in" \
        | ssh $SSH_OPTS "$fq" "cat > ~/.config/systemd/user/nnti-experiments.service" 2>/dev/null
      ssh $SSH_OPTS "$fq" "loginctl enable-linger ${USER_NAME} >/dev/null 2>&1;
                           systemctl --user daemon-reload;
                           systemctl --user reset-failed nnti-experiments.service 2>/dev/null;
                           systemctl --user enable --now nnti-experiments.service" 2>/dev/null
      echo "  ${h}: started (shard derived from hosts.txt)"
    ) &
    idx=$((idx+1)); [ $((idx % 6)) -eq 0 ] && wait
  done
  wait
}

# Both HPO phases install a unit that is byte-identical on every host: the
# journal on shared /home is the only coordination point, so workers need no
# per-host configuration and must not be given any (~/.config is NFS-shared and
# a per-host substitution is silently clobbered by whichever host wrote last).
hpo_launch() {
  local unit="$1" template="$2" trials="$3" epochs="$4"
  for h in $(hosts); do
    (
      fq="${h}.${DOMAIN}"
      ssh $SSH_OPTS "$fq" "mkdir -p ~/.config/systemd/user" 2>/dev/null
      sed -e "s|__TRIALS__|${trials}|" -e "s|__EPOCHS__|${epochs}|" \
          "${PROJECT}/deploy/${template}" \
        | ssh $SSH_OPTS "$fq" "cat > ~/.config/systemd/user/${unit}.service" 2>/dev/null
      ssh $SSH_OPTS "$fq" "loginctl enable-linger ${USER_NAME} >/dev/null 2>&1;
                           systemctl --user daemon-reload;
                           systemctl --user reset-failed ${unit}.service 2>/dev/null;
                           systemctl --user enable --now ${unit}.service" 2>/dev/null
      echo "  ${h}: ${unit} started (${trials} trials)"
    ) &
  done
  wait
}

hpo_stop() {
  for h in $(hosts); do
    ( ssh $SSH_OPTS "${h}.${DOMAIN}" \
        "for u in nnti-hpo nnti-hpo-peft nnti-hpo-lora; do
           systemctl --user stop \$u.service 2>/dev/null
           systemctl --user disable \$u.service 2>/dev/null
         done" >/dev/null 2>&1
      echo "  ${h}: hpo stopped" ) &
  done
  wait
}

stop() {
  for h in $(hosts); do
    ( ssh $SSH_OPTS "${h}.${DOMAIN}" \
        "systemctl --user stop nnti-experiments.service; systemctl --user disable nnti-experiments.service" \
        >/dev/null 2>&1; echo "  ${h}: stopped" ) &
  done
  wait
}

status() {
  printf "%-8s %-10s %-9s %-8s %s\n" HOST STATE DONE PENDING LAST
  for h in $(hosts); do
    (
      fq="${h}.${DOMAIN}"
      st=$(ssh $SSH_OPTS "$fq" "systemctl --user is-active nnti-experiments.service" 2>/dev/null)
      hb="${PROJECT}/experiments/heartbeat/${h}.json"
      if [ -f "$hb" ]; then
        read -r d p l < <(python3 -c "
import json;d=json.load(open('$hb'))
print(d.get('completed_this_session',0), d.get('pending_total',0), d.get('last_stage',''))" 2>/dev/null)
      else d=-; p=-; l=-; fi
      printf "%-8s %-10s %-9s %-8s %s\n" "$h" "${st:-?}" "$d" "$p" "$l"
    ) &
  done
  wait
  echo
  echo "total rows across hosts: $(cat ${PROJECT}/experiments/results/*.csv 2>/dev/null | grep -vc '^id,' || echo 0)"
}

case "${1:-}" in
  discover) discover ;;
  bootstrap) bootstrap ;;
  sync-artifacts) sync_artifacts ;;
  collect) collect ;;
  start) start ;;
  hpo) hpo_launch nnti-hpo nnti-hpo.service.in "${2:-12}" "${3:-15}" ;;
  hpo-peft) hpo_launch nnti-hpo-peft nnti-hpo-peft.service.in "${2:-4}" "${3:-12}" ;;
  hpo-lora) hpo_launch nnti-hpo-lora nnti-hpo-lora.service.in "${2:-5}" "${3:-12}" ;;
  hpo-stop) hpo_stop ;;
  stop) stop ;;
  status) status ;;
  *) sed -n '2,12p' "$0" ;;
esac

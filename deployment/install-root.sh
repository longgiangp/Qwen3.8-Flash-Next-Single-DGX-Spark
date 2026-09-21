#!/usr/bin/env bash
set -euo pipefail

[[ ${EUID:-$(id -u)} -eq 0 ]] || { echo "Run as root." >&2; exit 1; }
# Explicitly privileged installer for a trusted, reviewed checkout. Downloads
# nothing; does not open a network-facing inference listener.
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_BASE="/opt/qwen3.8-flash-next"
RELEASE_ID="mamba-cache-fix-v2"
RELEASE_DIR="$INSTALL_BASE/releases/$RELEASE_ID"
CURRENT_LINK="$INSTALL_BASE/current"
STATE_DIR="/var/lib/qwen3.8-flash-next"
MODEL_REVISION="925d7be6c14c6c9442ef83e8f05b5a3c39304f69"
MODEL_PATH="$STATE_DIR/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/$MODEL_REVISION"
IMAGE="vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
CORPUS_DIR="${QWEN_CORPUS_DIR:-$STATE_DIR/corpus}"
DRAFT_DIR="$STATE_DIR/draft_vocab"
DRAFT_VOCAB="$DRAFT_DIR/qwen38fn_local_code_65k.txt"
DRAFT_INPUT="${QWEN_DRAFT_VOCAB:-}"
UNIT=qwen3.8-flash-next.service
UNIT_FILE="/etc/systemd/system/$UNIT"
BACKUP_DIR="/var/backups/qwen3.8-flash-next/$(date -u +%Y%m%dT%H%M%SZ)-$RELEASE_ID-$$"
CUTOVER_STARTED=0
COMMITTED=0

# Writable installation parents must be root-owned, non-symlinks, and not
# writable by other users. HF snapshot files may contain standard cache links.
python3 - "$INSTALL_BASE/releases" "$STATE_DIR" "$DRAFT_DIR" /var/backups/qwen3.8-flash-next /etc/systemd/system <<'PY'
import pathlib, stat, sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    for item in reversed((path, *path.parents)):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise SystemExit(f"Unsafe installation directory: {item}")
PY
[[ ! -e "$CURRENT_LINK" || -L "$CURRENT_LINK" ]] || { echo "current must be a symlink, not a directory." >&2; exit 1; }
[[ ! -L "$UNIT_FILE" ]] || { echo "Refusing a symlinked unit file." >&2; exit 1; }
PREVIOUS_TARGET=""
if [[ -L "$CURRENT_LINK" ]]; then
    PREVIOUS_TARGET="$(readlink -f "$CURRENT_LINK" 2>/dev/null || true)"
    [[ "$PREVIOUS_TARGET" == "$INSTALL_BASE/releases/"* && -d "$PREVIOUS_TARGET" && ! -L "$PREVIOUS_TARGET" ]] || {
        echo "Current link must resolve to an existing release under $INSTALL_BASE/releases." >&2; exit 1;
    }
fi
if UNIT_LOAD="$(systemctl show "$UNIT" --property=LoadState --value)"; then
    :
else
    [[ "$UNIT_LOAD" == not-found ]] || { echo "Cannot inspect Qwen unit load state." >&2; exit 1; }
fi
case "$UNIT_LOAD" in loaded|not-found) ;; *) echo "Unexpected unit load state: $UNIT_LOAD" >&2; exit 1 ;; esac
if [[ "$UNIT_LOAD" == loaded && -z "$PREVIOUS_TARGET" ]]; then
    echo "An unmanaged Qwen unit already exists without a current release. Migrate it explicitly before using this installer." >&2
    exit 1
fi

rollback() {
    status=$?
    trap - EXIT INT TERM
    if [[ "$CUTOVER_STARTED" -eq 1 && "$COMMITTED" -eq 0 ]]; then
        echo "Cutover failed (status $status); stopping the candidate." >&2
        systemctl stop "$UNIT" 2>/dev/null || true
        if [[ -n "$PREVIOUS_TARGET" ]]; then
            ln -sfnT "$PREVIOUS_TARGET" "$CURRENT_LINK"
            if [[ -f "$BACKUP_DIR/$UNIT" ]]; then install -m 0644 "$BACKUP_DIR/$UNIT" "$UNIT_FILE"; fi
            systemctl daemon-reload
            systemctl reset-failed "$UNIT" 2>/dev/null || true
            systemctl start "$UNIT" || echo "Previous release did not restart; inspect its journal." >&2
        else
            # A failed first installation has no healthy release to restore.
            # Retain files for diagnosis, but never leave a broken boot start.
            systemctl disable "$UNIT" 2>/dev/null || true
            if [[ -L "$CURRENT_LINK" && "$(readlink "$CURRENT_LINK")" == "$RELEASE_DIR" ]]; then unlink "$CURRENT_LINK"; fi
            if [[ -f "$BACKUP_DIR/$UNIT" ]]; then install -m 0644 "$BACKUP_DIR/$UNIT" "$UNIT_FILE"; fi
            systemctl daemon-reload
            echo "No previous managed release exists. Candidate remains stopped and disabled." >&2
        fi
    fi
    exit "$status"
}
trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_existing_release() {
    python3 "$STAGE_DIR/deployment/manifest_release.py" verify "$STAGE_DIR" "$RELEASE_DIR"
    [[ -f "$RELEASE_DIR/.env" && ! -L "$RELEASE_DIR/.env" ]] || { echo "Existing release has no ordinary .env file; use a new release ID." >&2; return 1; }
}
wait_for_ready() {
    local unit_state
    for _ in $(seq 1 180); do
        if curl -fsS --max-time 3 http://127.0.0.1:8888/health >/dev/null 2>&1; then return 0; fi
        unit_state="$(systemctl show "$UNIT" --property=ActiveState --value)"
        case "$unit_state" in active|activating|reloading) ;; *) echo "Qwen stopped before readiness (state=$unit_state)." >&2; return 1 ;; esac
        sleep 10
    done
    echo "Qwen did not become healthy within 30 minutes." >&2
    return 1
}

cd "$STAGE_DIR"
python3 deployment/manifest_release.py verify "$STAGE_DIR" "$STAGE_DIR"
/bin/bash deployment/check-host-profile.sh
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors.index.json" ]] || { echo "Pinned HF model is absent. Complete deployment/README.md bootstrap first." >&2; exit 1; }
printf '%s  %s\n' \
    "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e" "$MODEL_PATH/config.json" \
    "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9" "$MODEL_PATH/model.safetensors.index.json" | sha256sum --check -
docker image inspect "$IMAGE" >/dev/null || { echo "Pull the pinned image explicitly first; see deployment/README.md." >&2; exit 1; }
if [[ "$PREVIOUS_TARGET" == "$RELEASE_DIR" ]]; then
    verify_existing_release
    echo "Release already current; preserving .env and rollback history. New settings require a new release ID."
    systemctl reset-failed "$UNIT" 2>/dev/null || true
    systemctl start "$UNIT"
    wait_for_ready
    echo "Qwen release is healthy: $RELEASE_DIR"
    exit 0
fi

# A prebuilt vocabulary or explicit full-vocabulary bootstrap avoids requiring
# any private corpus. Neither corpus nor token frequencies are uploaded.
if [[ -e "$RELEASE_DIR" || -L "$RELEASE_DIR" ]]; then
    verify_existing_release
    saved_draft="$(/bin/bash -c 'source "$1"; printf "%s\n" "${MTP_DRAFT_VOCAB:-}"' bash "$RELEASE_DIR/.env")"
    if [[ -z "$saved_draft" ]]; then
        [[ -z "$DRAFT_INPUT" || "$DRAFT_INPUT" == off ]] || { echo "Existing release disables reduced vocabulary; use a new release ID to change it." >&2; exit 1; }
        DRAFT_INPUT=off
    elif [[ "$DRAFT_INPUT" == off ]]; then
        echo "Existing release enables reduced vocabulary; use a new release ID to change it." >&2; exit 1
    fi
fi
if [[ "$DRAFT_INPUT" == off ]]; then
    DRAFT_VOCAB=""
elif [[ -n "$DRAFT_INPUT" ]]; then
    [[ "$DRAFT_INPUT" == /* && -f "$DRAFT_INPUT" && ! -L "$DRAFT_INPUT" ]] || { echo "QWEN_DRAFT_VOCAB must be 'off' or an absolute ordinary file." >&2; exit 1; }
elif [[ ! -s "$DRAFT_VOCAB" ]]; then
    [[ "$CORPUS_DIR" == /* && "$CORPUS_DIR" != *:* && -d "$CORPUS_DIR" && ! -L "$CORPUS_DIR" && -s "$CORPUS_DIR/local_code.txt" && -s "$CORPUS_DIR/model_outputs.jsonl" ]] || {
        echo "Provide local_code.txt and model_outputs.jsonl under QWEN_CORPUS_DIR ($CORPUS_DIR), an existing QWEN_DRAFT_VOCAB, or QWEN_DRAFT_VOCAB=off. See deployment/README.md." >&2; exit 1;
    }
fi
/bin/bash "$STAGE_DIR/deployment/assert-idle.sh"
install -d -m 0755 "$INSTALL_BASE/releases" "$STATE_DIR" "$DRAFT_DIR"
install -d -m 0700 "$BACKUP_DIR"
if [[ -f "$UNIT_FILE" ]]; then install -m 0600 "$UNIT_FILE" "$BACKUP_DIR/$UNIT"; fi
printf '%s\n' "$PREVIOUS_TARGET" > "$BACKUP_DIR/previous-release"
if [[ -e "$RELEASE_DIR" || -L "$RELEASE_DIR" ]]; then
    verify_existing_release
    echo "Existing candidate keeps its .env. Use a new release ID to change configuration."
else
    python3 deployment/manifest_release.py copy "$STAGE_DIR" "$RELEASE_DIR"
    install -m 0644 "$RELEASE_DIR/.env.sample" "$RELEASE_DIR/.env"
    printf '\n# Selected explicitly at installation; no runtime downloads.\nMTP_DRAFT_VOCAB=%q\n' "$DRAFT_VOCAB" >> "$RELEASE_DIR/.env"
fi
# An existing immutable release keeps its actual saved vocabulary selection.
saved_draft="$(/bin/bash -c 'source "$1"; printf "%s\n" "${MTP_DRAFT_VOCAB:-}"' bash "$RELEASE_DIR/.env")"
[[ "$saved_draft" == "$DRAFT_DIR/qwen38fn_local_code_65k.txt" || -z "$saved_draft" ]] || { echo "Unexpected saved vocabulary path; audit a new release." >&2; exit 1; }
DRAFT_VOCAB="$saved_draft"
container_mem_gib="$(/bin/bash -c 'source "$1"; printf "%s\n" "${CONTAINER_MEM_GIB:-}"' bash "$RELEASE_DIR/.env")"
[[ "$container_mem_gib" =~ ^[1-9][0-9]*$ ]] || { echo "CONTAINER_MEM_GIB must be an explicit positive integer." >&2; exit 1; }
required_available_kib=$(( (container_mem_gib + 4) * 1048576 ))
CUTOVER_STARTED=1
# Ignore only an absent unit, not failed stop or permission errors.
if [[ "$UNIT_LOAD" != not-found ]]; then systemctl stop "$UNIT"; fi
for _ in $(seq 1 120); do
    [[ -z "$(ss -H -ltn 'sport = :8888')" ]] && break
    sleep 2
done
[[ -z "$(ss -H -ltn 'sport = :8888')" ]] || { echo "Port 8888 remains occupied after stopping Qwen." >&2; exit 1; }
for _ in $(seq 1 120); do
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    [[ "$available_kib" -ge "$required_available_kib" ]] && break
    sleep 2
done
[[ "$available_kib" -ge "$required_available_kib" ]] || { echo "Memory did not recover to $((container_mem_gib + 4)) GiB." >&2; exit 1; }
if [[ -n "$DRAFT_VOCAB" ]]; then
    [[ ! -L "$DRAFT_VOCAB" ]] || { echo "Refusing symlinked draft vocabulary destination." >&2; exit 1; }
    if [[ -n "$DRAFT_INPUT" && "$DRAFT_INPUT" != "$DRAFT_VOCAB" ]]; then
        [[ ! -e "$DRAFT_VOCAB" ]] || { echo "Draft vocabulary already exists; refusing replacement. Keep it or version its path in a new release." >&2; exit 1; }
        python3 "$RELEASE_DIR/deployment/validate_draft_vocab.py" "$DRAFT_INPUT" "$MODEL_PATH"
        install -m 0644 "$DRAFT_INPUT" "$DRAFT_VOCAB"
    elif [[ ! -s "$DRAFT_VOCAB" ]]; then
        docker run --rm --pull=never --network none --memory 8g --memory-swap 8g --cpus 16 \
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e DO_NOT_TRACK=1 \
            -v "$STATE_DIR/huggingface:/hf:ro" -v "$CORPUS_DIR:/corpus:ro" \
            -v "$RELEASE_DIR/files/build_draft_vocab.py:/builder.py:ro" -v "$DRAFT_DIR:/out" \
            --entrypoint python3 "$IMAGE" /builder.py /corpus/local_code.txt /corpus/model_outputs.jsonl:20 \
            --model "/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/$MODEL_REVISION" \
            --size 65536 --out /out/qwen38fn_local_code_65k.txt
    fi
    python3 "$RELEASE_DIR/deployment/validate_draft_vocab.py" "$DRAFT_VOCAB" "$MODEL_PATH"
fi
ln -sfnT "$RELEASE_DIR" "$CURRENT_LINK"
install -m 0644 "$RELEASE_DIR/deployment/$UNIT" "$UNIT_FILE"
systemctl daemon-reload
systemctl enable "$UNIT"
systemctl reset-failed "$UNIT" 2>/dev/null || true
QWEN_STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
systemctl start "$UNIT"
wait_for_ready
models="$(curl -fsS --max-time 15 http://127.0.0.1:8888/v1/models)"
python3 -c 'import json,sys; d=json.load(sys.stdin); assert any(m.get("id")=="qwen3.8-flash-next" and m.get("max_model_len")==262144 for m in d["data"])' <<< "$models"
if journalctl -k --since "$QWEN_STARTED" -q | grep 'NV_ERR_NO_MEMORY' >/dev/null; then echo "NVIDIA allocation errors appeared during launch." >&2; exit 1; fi
printf '%s\n' "$PREVIOUS_TARGET" > "$STATE_DIR/previous-release"
printf '%s\n' "$RELEASE_DIR" > "$STATE_DIR/current-release"
COMMITTED=1
trap - EXIT INT TERM
echo "Qwen cutover completed: $RELEASE_DIR"
echo "Loopback API: http://127.0.0.1:8888/v1 (configure authenticated access separately)."
echo "Backup: $BACKUP_DIR"

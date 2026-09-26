# Pinned, loopback-only systemd deployment

This optional deployment layer packages the measured profile as an immutable
release with memory admission checks, readiness checks, bounded restart attempts,
and rollback. It is independent of any particular laptop, user account, reverse
proxy, or VPN. The optimized production profile and the generalized public installer were
exercised end to end on one NVIDIA GB10 DGX Spark on 2026-09-09. This is one-host
evidence, not portability proof for every DGX OS, driver, image, or model revision.

## Before running privileged code

- Use a single 128 GB DGX Spark running a supported NVIDIA DGX OS. Docker with
  the NVIDIA Container Toolkit/GPU runtime, `nvidia-smi`, systemd, Bash, Python 3,
  `curl`, `iproute2` (`ss`), GNU coreutils, and `sha256sum` must already work.
  Install those prerequisites through NVIDIA's official DGX documentation and
  your OS package sources. These scripts do not install Docker or drivers.
- Allow at least 150 GiB for the model snapshot, packed PLE table, image/cache
  growth, plus the Docker image's actual size and ordinary OS free space.
  Image layers and temporary build files can require substantial additional
  space. Check both the model-state and Docker filesystems before downloading.
- Review this repository and its pinned image/model provenance first. A digest
  pins bytes; it is not a malware scan, vendor endorsement, or cryptographic
  signature from this fork. The checksum manifest protects against accidental
  payload drift; it does not make an untrusted checkout safe to execute as root.
- Do not let another process edit the checkout while installation is running.
  Runtime `.env` is sourced as shell code and must remain root-owned and trusted.
  Do not put tokens in this public checkout or publish local model output/corpora.
- Stop other GPU-serving processes yourself. The installer will refuse to
  interrupt active or queued requests on port 8888. Drain incoming traffic at
  your proxy before cutover: the idle probe is a point-in-time check, not an
  atomic admission lock.
- An existing service with the same unit name but no managed `current` release
  is not silently replaced. Back up and explicitly migrate that installation
  first. Failed first-install files/units are retained for diagnosis and likewise
  require an explicit reviewed recovery step before retrying.

The model is the **third-party MiaAI NVFP4 quantization hosted on Hugging Face**,
not an official Qwen-authored quantization. This profile keeps the stock checkpoint;
it does not download or enable the optional abliterated variant.

## 1. Explicitly bootstrap the pinned image and HF snapshot

Run from the reviewed repository root. These are the only download steps here:
the image comes from its Docker registry, and model files come from Hugging Face
and its official storage/CDN infrastructure. No alternate model mirrors are used.

```bash
sudo docker pull vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
sudo install -d -m 0755 /var/lib/qwen3.8-flash-next/huggingface
sudo docker run --rm --pull=never \
  -e HF_HOME=/hf -e HF_ENDPOINT=https://huggingface.co \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e DO_NOT_TRACK=1 \
  -v /var/lib/qwen3.8-flash-next/huggingface:/hf \
  --entrypoint python3 \
  vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8 \
  -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4", revision="925d7be6c14c6c9442ef83e8f05b5a3c39304f69", cache_dir="/hf/hub")'
```

This stock snapshot does not need the abliterated model's gated access. Do not
accept another model's access or contact-sharing terms as part of this procedure.
The bootstrap above is resumable. The installer checks pinned config/index
hashes; the runtime checks all shards named by the index are present. Those are
not a separate hash audit of every model weight shard.

## 2. Choose a draft-vocabulary mode

First review the system-wide VM profile in
[`files/sysctl-spark3.conf`](../files/sysctl-spark3.conf). The shipped 4 GiB
watchdog floors are qualified only with the exact settings below; both the
installer and each systemd startup refuse mismatches. These are explicit opt-in
operator commands, not automatic installer changes:

```bash
sysctl vm.min_free_kbytes vm.watermark_scale_factor vm.swappiness
sudo sysctl -p files/sysctl-spark3.conf
sudo install -b -m 0644 files/sysctl-spark3.conf /etc/sysctl.d/90-qwen38-qualified-profile.conf
```

The first `sudo` command applies the reviewed values immediately; the second
persists them and backs up an existing destination using GNU `install -b`.
Resolve any conflicting sysctl files and verify values again after reboot.
Expected values are `4194304`, `300`, and `30`, respectively. These settings
affect the entire machine and memory accounting, not only this container.

The measured profile uses a locally built reduced draft vocabulary, retaining
55,124 token IDs in the evaluated corpus. No private corpus or vocabulary is
included in this fork. The requested maximum is 65,536, not a guaranteed output
count: the builder retains observed tokens plus all added/special tokens.

**Representative local corpus (recommended for reproducing the optimization):**
prepare a directory outside this repository containing:

- `local_code.txt`: code/text you are authorized to process locally.
- `model_outputs.jsonl`: one JSON object per line, with a `text` field containing
  representative model-generated coding answers. Do not use only prompts: MTP
  predicts the model's outputs. The builder weights this file 20 times.

For example, use an editor to put a harmless representative code excerpt in
`local_code.txt` and write actual, locally collected output records in this form:

```json
{"text":"def add(a, b):\n    return a + b\n"}
```

That single illustrative record is **not** an adequate performance corpus. Use
diverse representative languages, formatting, tool arguments, and task outputs.
The installer builds with the pinned local tokenizer in a network-disabled
container. Corpus and HF model mounts are read-only; only the vocabulary output
directory is writable. It validates unique/ranged IDs and retention of tokenizer
added/special tokens. Coverage and draft acceptance should be remeasured on your
workload; the private evaluation corpus is deliberately not published.

```bash
sudo env QWEN_CORPUS_DIR=/absolute/path/to/your/local-corpus \
  /bin/bash deployment/install-root.sh
```

The default corpus directory is `/var/lib/qwen3.8-flash-next/corpus`. Files are
read locally only; the installer never searches your home directory.

**Existing vocabulary:** supply an ordinary token-ID file from the pinned
tokenizer. It is validated and copied into the generic state directory, not into
the Git checkout. Existing vocabulary files are not overwritten.

```bash
sudo env QWEN_DRAFT_VOCAB=/absolute/path/to/draft-vocabulary.txt \
  /bin/bash deployment/install-root.sh
```

**Bootstrap without a corpus:** use the full draft vocabulary. This remains MTP3,
but disables the reduced-vocabulary optimization, so the measured throughput
table does **not** apply directly.

```bash
sudo env QWEN_DRAFT_VOCAB=off /bin/bash deployment/install-root.sh
```

The first model launch can take 10–30 minutes, especially when building the PLE
table and warming the GPU. Installer readiness allows approximately 30 minutes.

## Installed layout and memory policy

| Item | Location / behavior |
| --- | --- |
| Immutable source release | `/opt/qwen3.8-flash-next/releases/mamba-cache-fix-v4` |
| Active link | `/opt/qwen3.8-flash-next/current` |
| Model/cache/PLE state | `/var/lib/qwen3.8-flash-next` |
| Vocabulary | `/var/lib/qwen3.8-flash-next/draft_vocab/qwen38fn_local_code_65k.txt` |
| Root-only rollback backup | `/var/backups/qwen3.8-flash-next/<timestamp>-<release>-<pid>` |
| API | `http://127.0.0.1:8888/v1` |
| Model ID | `qwen3.8-flash-next` |
| Context / active scheduler slots | 262,144 tokens / 8 short requests |
| Host reserve / container cap | 36 GiB / 90 GiB |
| Launch admission | At least 94 GiB `MemAvailable` |
| Automatic restart policy | Initial start plus at most one retry per hour; 60-second delay |

The 8 scheduler slots do **not** mean eight simultaneous full-context requests
fit. The measured KV pool was 5.41 GiB / 360,264 tokens; actual pool size can vary
with host/runtime memory accounting. Long prefill also competes with decoding.
Keep the watchdog thresholds; do not lower them to force an unstable launch.
The measured host used a persistent 4 GiB kernel reserve, watermark scale 300,
and swappiness 30. This installer refuses mismatching kernel sysctls rather than
silently changing them; explicitly review/apply the host profile above and
requalify your machine. `MemAvailable` is not directly comparable across differing
kernel reserve policies.

Only manifest-listed ordinary files are installed. `.git`, private `.env` files,
unlisted logs, credentials, and corpora are not copied. A checked-in `.env.sample`
creates the runtime `.env`; existing immutable releases preserve their own copy.
Change release IDs and regenerate the manifest after audited source/config
changes instead of mutating an existing release in place. The generated runtime
`.env` is intentionally outside the source checksum manifest.

## KV pool is pinned

The release pins the KV cache at 5 GiB (`KV_CACHE_MEMORY` in `.env.sample`, passed as
`--kv-cache-memory-bytes`). Left to vLLM, the pool is derived from a startup memory profile
that varied between 3.54 and 9.32 GiB across four launches of an identical configuration on
one GB10 host; one launch fell below the 3.92 GiB a single 262,144-token request needs and
vLLM refused to start. Do not remove the pin without repeating that comparison.

## Readiness, access, and rollback

```bash
systemctl status qwen3.8-flash-next.service
journalctl -u qwen3.8-flash-next.service -n 80 --no-pager
curl --fail http://127.0.0.1:8888/health
curl --fail http://127.0.0.1:8888/v1/models
```

No gateway/VPN installation or credentials are assumed. For another computer,
use an SSH tunnel or separately configure an authenticated HTTPS reverse proxy
over a private network. Keep the model server bound to loopback; do not expose
this unauthenticated HTTP backend directly to the LAN or internet. An
OpenAI-compatible client that expects the complete route uses your separately configured
proxy's `/v1/chat/completions` URL and the model ID above. Test that unauthorized
requests fail and streaming/tool calls work through that proxy independently.

Failed upgrades restore and start the previous managed release. A failed first
installation has no prior release: the candidate is stopped and disabled, its
current link removed, and files retained for diagnosis. For an explicit rollback
after a successful upgrade, first drain requests and run:

```bash
sudo /bin/bash /opt/qwen3.8-flash-next/current/deployment/rollback-root.sh
```

Rollback requires a previous managed release and starts it; check readiness
afterward. Historical files are retained rather than deleting models/releases.

## Optional OpenWebUI Docker proxy

The production runner deliberately requires vLLM to remain on
`127.0.0.1:8888`. If an existing Dockerized OpenWebUI connection already uses
`http://host.docker.internal:8000/v1`, install the managed socket proxy:

```bash
sudo bash ./install-openwebui-proxy-root.sh
```

The installer discovers the IPv4 address assigned to `docker0` and listens only
on that address. It refuses an absent Docker bridge, an arbitrary listen address,
an invalid port, and an unmanaged or partial pre-existing unit. It does not edit
OpenWebUI, the Qwen runtime `.env`, or the loopback-only production guard. The
request path is preserved:

```text
OpenWebUI -> docker0:8000 -> 127.0.0.1:8888 -> vLLM
```

Verify both ends after Qwen reports `Application startup complete`:

```bash
curl --fail http://127.0.0.1:8888/v1/models
docker exec open-webui curl --fail \
  http://host.docker.internal:8000/v1/models
```

Port 8000 is the default only because it matches the stated existing OpenWebUI
connection. Override ports explicitly when necessary:

```bash
sudo bash ./install-openwebui-proxy-root.sh \
  --listen-port 8000 --backend-port 8888
```

Remove only the proxy, without stopping or deleting Qwen or OpenWebUI:

```bash
sudo bash ./uninstall-openwebui-proxy-root.sh
```

The main `uninstall-root.sh` also removes these managed proxy units by default.
Both uninstall paths refuse same-named unit files that do not carry this recipe's
management marker.

## Uninstall

Run the uninstaller from a reviewed Git checkout. Its default removes the managed
service, exact container, managed OpenWebUI proxy units, systemd unit, and immutable releases while preserving
the expensive model cache, state, rollback backups, pinned image, host sysctl
profile, and the checkout itself:

```bash
sudo bash ./uninstall-root.sh
```

The script prints its exact plan and requires typing `REMOVE`. For automation,
add `--yes` only after reviewing that plan. Optional destructive scopes are
independent: `--purge-state`, `--purge-backups`, `--remove-image`, and
`--remove-sysctl-profile`; `--purge-all` selects all four. Removing the sysctl
file does not guess the machine's former live values. Reboot or apply a separately
reviewed host policy afterward. Use `sudo bash ./uninstall-root.sh --help` before any
purge operation.

## Optional OS maintenance

`install-root.sh` does not update the OS. During a planned outage, review
`sudo apt-get -s dist-upgrade`, drain traffic, and then explicitly run:

```bash
sudo /bin/bash deployment/full-maintenance-root.sh
```

This wrapper requires an existing service, stops inference before package work,
updates APT/snap, refreshes available firmware metadata, and invokes the installer.
It does not force phased updates, autoremove old kernels, install firmware, or
reboot. On failure it attempts to restart the current service. The standalone
`maintenance-root.sh` must not be run alongside a memory-heavy model. Review
APT's proposed package changes before accepting a maintenance outage.

## Local validation

```bash
python3 -m unittest discover -s deployment/tests -v
python3 -m unittest discover -s bench -p 'test_*.py' -v
python3 deployment/manifest_release.py verify . .
```

These cover manifest isolation/path validation and vocabulary validation. A
2026-09-09 GB10 run additionally covered the pinned download, host profile,
privileged systemd install, smoke test and mixed load; future dependency or host
changes still require requalification. A subsequent 20-repeat 32K smoke run
kept greedy text and first-token logprobs stable, increased prefix hits by
29,952 tokens, and measured 18.809/1.848-second cold/warm TTFT.

On a running loopback-bound backend, execute the non-destructive behavioral
validation before considering image-level performance patches:

```bash
python3 bench/runtime_validation.py smoke
python3 bench/runtime_validation.py mixed --prefill-tokens 32768
```

The smoke test checks coherence, repeated greedy output/logprob stability, long
prompt TTFT and the prefix-cache metric when available. The mixed test measures
decode stream gaps while a long prefill competes with it. Neither command changes
the server configuration.

### Prefix-cache correctness and QSA determinism

The ordinary smoke test proves latency and one short-prompt repetition, but that
alone cannot exclude a wrong Mamba-state restore on a cache hit or a
shape-dependent GB10 sparse-attention top-k variation. The stronger validator
compares generated-answer SHA-256 values and first-token top-logprobs without
saving generated text:

```bash
python3 bench/cache_correctness.py all \\
  --cache-sizes 8192,32768,131072 \\
  --qsa-sizes 0,8192,32768 \\
  --repeats 5 --max-tokens 96 --require-prefix-hit \\
  --output qwen38-cache-qsa-$(date -u +%Y%m%dT%H%M%SZ).json
```

Each cache case uses a unique early marker, runs a first observation and an
immediate repeat, requires the prefix-hit metric to increase, and compares the
full answer hash plus first-token scores.

**Current pinned-image result (2026-09-09): failed.** Zero-context cases were
stable, while long-context cache/QSA answer hashes or first-token logprobs varied.
Do not treat the working prefix-hit counter as correctness proof. See
[`bench/results/2026-09-09-cache-qsa-validation.json`](../bench/results/2026-09-09-cache-qsa-validation.json). QSA cases cover four instruction types
at three context sizes. A pass is evidence for this pinned image, prompt suite
and host; repeat after any model, image, CUDA kernel, cache-policy or scheduler
change. A failure must be investigated before importing an external patch.

### Storage and prefill observation

The observer runs real-text first-observed and immediate-repeat prefills at 8K,
32K and 128K by default. Each size receives an early unique marker so one size
does not reuse another size's long prefix. It records vLLM prefix hits,
`/proc/vmstat`, selected `/proc/meminfo` values, readable vLLM process-tree
fault/I/O counters, physical block-device read counters and packed PLE table
sizes. It never invokes `drop_caches`, changes a sysctl, or restarts the service;
therefore "first-observed" must not be described as a guaranteed cold OS cache.

```bash
python3 bench/storage_prefill.py \\
  --sizes 8192,32768,131072 \\
  --output qwen38-storage-prefill-$(date -u +%Y%m%dT%H%M%SZ).json
```

Measured on the already-running production service without clearing caches:

| Prompt | First TTFT | Immediate repeat | Prefix-hit increase | First global major faults | First NVMe reads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 4.324 s | 1.882 s | 4,992 tokens | 0 | 6.51 MB |
| 32K | 17.408 s | 1.841 s | 29,952 tokens | 0 | 0.34 MB |
| 128K | 73.824 s | 2.277 s | 128,128 tokens | 149 | 9.03 MB |

This warm-service observation found no material SSD paging bottleneck. Global
VM/block counters can include unrelated activity, and process `read_bytes` does
not reliably attribute mmap-fault I/O. Preserve the raw local report for deeper
host diagnosis; the checked-in summary removes host paths and PIDs.

Run it as the ordinary user first. Some kernels restrict another user's
`/proc/<pid>/io`; unavailable process counters remain null/empty while global
kernel and block-device counters are still reported. Do not use sudo merely to
make optional counters appear unless the script and output path have been
reviewed. The report contains the absolute corpus and PLE paths but no generated
answer text.

## Clone, install, test and collect one log

From an empty working directory on the DGX Spark, download the orchestration
script and first run its non-mutating preflight:

```bash
curl -fL -o clone-install-test.sh \
  https://raw.githubusercontent.com/kjlee-inlct/Qwen3.8-Flash-Next-Single-DGX-Spark/main/clone-install-test.sh
chmod +x clone-install-test.sh
./clone-install-test.sh
```

After reviewing the preflight log, the explicit full mode clones the validation
branch, pulls the pinned image, downloads the pinned stock checkpoint, applies
the documented system-wide VM profile, installs the service, and runs both
behavioral tests:

```bash
./clone-install-test.sh --full
```

Full mode uses `HF_TOKEN` when it is already exported. If it is unset, the
script asks for an optional token with terminal echo disabled; pressing Enter
continues anonymously. A literal token command-line option is intentionally not
provided because command arguments can remain in shell history and process
listings. For unattended anonymous execution, add `--no-hf-token-prompt`.

The default target is `./Qwen3.8-Flash-Next-Single-DGX-Spark`. Use
`--target /absolute/path` to change it. An existing target is refused unless it
is the expected clean checkout at the requested branch and `--reuse` is supplied.
Full mode defaults to the complete MTP vocabulary so no private corpus is needed;
this is functional bootstrap mode and does not reproduce the fitted-vocabulary
performance measurement. Supply `--draft-vocab /absolute/file` or
`--corpus-dir /absolute/directory` to install a workload-specific vocabulary.

The final line prints the absolute log path. Send that single log back for review.
The script never enables shell tracing and never prints `HF_TOKEN`.

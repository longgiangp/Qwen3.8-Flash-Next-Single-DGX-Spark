# Third-party code

## files/mamba_utils_guarded.py

vLLM's `vllm/v1/worker/mamba_utils.py` (Apache-2.0, Copyright contributors to
the vLLM project) with two changes, taken unmodified from
`blazux/qwen3.8-Flash-DGX` `src/mamba_utils_guarded.py`
(identical in `techfury90/qwen3.8-Flash-DGX`; both Apache-2.0):

- vllm#50729 "[Bugfix][Mamba] Fix overlapping state copy race" (@AndreasKaratzas):
  a conv-state copy whose source and destination are the same block
  (`src == dest`, `token_bias > 0`) was an unsafe overlapping memcpy.
- an out-of-range block-id guard (@Saren-Arterius): such a copy is skipped and
  counted ("mamba state-copy guard: N out-of-range ...") instead of faulting.

The file is applied only when the image's original `mamba_utils.py` hashes to
`MAMBA_UTILS_ORIG_SHA256` in `start.sh`; any other image aborts the launch.

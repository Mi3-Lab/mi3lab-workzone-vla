## Title

[Bug] Checkpoint loader silently overwrites a trained `lm_head.weight` with `embed_tokens.weight` when `tie_word_embeddings=true` but both tensors are present and differ

## Summary

`checkpoint/loader.py`'s tied-embeddings post-processing step
(`_load_checkpoint`, around
[loader.py:182-194](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/tensorrt_edgellm/checkpoint/loader.py#L182-L194))
unconditionally calls `model.tie_weights()` whenever the HF `config.json`
declares `tie_word_embeddings: true` and the model's `lm_head` is an
`FP16Linear`:

```python
config = getattr(model, "config", None)
if (hasattr(model, "tie_weights") and config is not None
        and getattr(config, "tie_word_embeddings", False)):
    from ..models.linear import FP16Linear
    if isinstance(getattr(model, "lm_head", None), FP16Linear):
        model.tie_weights()
        logger.info("Tied lm_head.weight to embed_tokens.weight")
```

This assumes `tie_word_embeddings=true` implies the checkpoint has no
independent `lm_head.weight` tensor (the standard HF convention: tied
models omit `lm_head.weight` from the saved state dict entirely, relying on
`tie_weights()` to restore the share at load time). But if a checkpoint
**does** ship an explicit `lm_head.weight` that has diverged from
`embed_tokens.weight` (e.g. because a fine-tune trained the head separately
even though the base model's config still says `tie_word_embeddings:
true`), this code path discards the loaded, trained `lm_head.weight` and
silently replaces it with the input embedding matrix — with no warning,
even though both tensors were present in the checkpoint with different
values.

By contrast, `transformers.PreTrainedModel.from_pretrained` (used for the
same checkpoint in our validation) detects this exact situation and
**skips the tie**, emitting:

```
The tied weights mapping and config for this model specifies to tie
model.language_model.embed_tokens.weight to lm_head.weight, but both are
present in the checkpoints with different values, so we will NOT tie
them. You should update the config with `tie_word_embeddings=False` to
silence this warning.
```

## Impact

For a 2B Qwen3-VL-derived VLM fine-tuned on a downstream task (our
checkpoint), this produced severe, hard-to-diagnose generation quality
degradation *only* in the TensorRT-Edge-LLM runtime — the HF/PyTorch and
ModelOpt fake-quant paths were unaffected (because `transformers` correctly
skipped the tie), which made it look at first like an INT4/AWQ
quantization artifact rather than a checkpoint-loading bug.

Symptom: the model's own `<|answer_start|>`-equivalent special token
(id 155695, added during our SFT for chat formatting) suffered the largest
logit corruption from the wrong `lm_head`, since it is exactly the token
whose row diverged most between the trained `lm_head.weight` and
`embed_tokens.weight` (cosine similarity ~0.79 vs. ~0.98+ for ordinary
vocabulary tokens). This made a rare/low-frequency token win the very
first post-prefill sampling step in **92% of 685 real end-to-end
requests** (`.DATA`, `.INSTANCE`, and similar tokens), occasionally
cascading into repetition loops or off-domain hallucinated continuations
in longer generations.

We root-caused this by comparing the `lm_head`-producing `MatMul`
initializer in the exported `model.onnx` against both
`lm_head.weight` and `embed_tokens.weight` from the source HF checkpoint,
row by row (fp32 upcast, per-token-id max-abs-diff):

```
token 155695: onnx-vs-EMBED maxdiff=0.000000  |  onnx-vs-LMHEAD maxdiff=0.019562
token  52728: onnx-vs-EMBED maxdiff=0.000000  |  onnx-vs-LMHEAD maxdiff=0.027222
token    100: onnx-vs-EMBED maxdiff=0.000000  |  onnx-vs-LMHEAD maxdiff=0.047211
```

i.e. the ONNX weight is byte-identical to `embed_tokens.weight`, not
`lm_head.weight`, confirming the tie was applied when it should not have
been.

## Fix applied on our side (workaround, not a code change to this repo)

Setting `"tie_word_embeddings": false` explicitly in the checkpoint's
`config.json` before running `tensorrt-edgellm-quantize` /
`tensorrt-edgellm-export` avoids the erroneous `tie_weights()` call
entirely. After re-exporting and rebuilding the engine with this config
change (no other change), the same 685-request batch went from:

| Metric | Before | After |
|---|---|---|
| Response starts with a low-frequency/junk token | 92% | 0% |
| Long-generation repetition loops | 8% | 0% |
| Long-generation off-domain hallucination | 25% | 0% |

## Suggested fix

Mirror `transformers`' safety check in
`checkpoint/loader.py` before calling `model.tie_weights()`: if
`lm_head.weight` was already loaded from the checkpoint (i.e. present in
the shard map / state dict) **and** differs materially from
`embed_tokens.weight`, skip the tie and keep the loaded `lm_head.weight`
(optionally emit a warning, same as `transformers` does). Only fall back
to tying when the checkpoint omits `lm_head.weight` entirely, which is the
case the current code seems to assume is universal.

## Environment

- TensorRT-Edge-LLM: cloned from `main`, built via NGC container
  `nvcr.io/nvidia/tensorrt:24.08-py3` (TensorRT 10.3.0, CUDA 12.6)
- Quantization: `tensorrt_edgellm.quantization` INT4 AWQ (`int4_awq`
  curated recipe), ModelOpt 0.44.0
- Base architecture: `qwen3_vl` / `Qwen3VLForConditionalGeneration`
  (Cosmos-Reason2-2B lineage), vocab_size 155697, custom SFT-added special
  tokens (`<|answer_start|>`/`<|answer_end|>`)
- Hardware for engine build/inference: NVIDIA A100 (compute capability 8.0)

## Reproduction (minimal)

1. Take any HF checkpoint where `config.json` has `tie_word_embeddings:
   true` but the safetensors shards contain both `lm_head.weight` and
   `model.embed_tokens.weight` (or `model.language_model.embed_tokens.weight`)
   with different values — e.g. any base model with tied embeddings that
   was then fine-tuned with an untied head without updating the config.
2. Run `tensorrt-edgellm-quantize` / `tensorrt-edgellm-export` and build the
   engine.
3. Compare the exported ONNX's final `MatMul` weight (the `lm_head`
   projection) against the checkpoint's `lm_head.weight` tensor — it will
   match `embed_tokens.weight` instead.

Happy to share the diagnostic scripts (weight-comparison script + the
685-request batch harness) if useful for a regression test.

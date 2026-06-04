# Guidance Scale NPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a command-line `--guidance-scale` option to the NPU entrypoint so image generation guidance can be set to `1.0` without editing code.

**Architecture:** Keep the change scoped to the NPU runner and its tests. The CLI should accept a floating-point guidance scale, pass it through `generate_image()`, and leave model internals unchanged.

**Tech Stack:** Python, argparse, pytest

---

### Task 1: Add CLI plumbing

**Files:**
- Modify: `run_image_gen_npu.py`

- [ ] **Step 1: Write the failing test**

```python
def test_guidance_scale_defaults_to_model_value(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_image_gen_npu.py", "--prompt", "p"])
    args = parse_args()
    assert args.guidance_scale is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_run_image_gen_npu.py::test_guidance_scale_defaults_to_model_value -v`
Expected: FAIL with `AttributeError` because `guidance_scale` is not defined yet.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument(
    "--guidance-scale",
    type=float,
    default=None,
    help="Override the diffusion guidance scale. Use 1.0 to disable extra guidance.",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_run_image_gen_npu.py::test_guidance_scale_defaults_to_model_value -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run_image_gen_npu.py tests/test_run_image_gen_npu.py
git commit -m "feat: add guidance scale cli option"
```

### Task 2: Pass guidance scale through to generation

**Files:**
- Modify: `run_image_gen_npu.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_model_kwargs_keeps_guidance_scale_out_of_model_load():
    args = SimpleNamespace(
        attn_impl="sdpa",
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
        guidance_scale=1.0,
    )
    kwargs = build_model_kwargs(args)
    assert "guidance_scale" not in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_run_image_gen_npu.py::test_build_model_kwargs_keeps_guidance_scale_out_of_model_load -v`
Expected: FAIL until the test exists and documents the intended separation of concerns.

- [ ] **Step 3: Write minimal implementation**

```python
cot_text, samples = model.generate_image(
    prompt=prompt,
    seed=args.seed,
    image_size=args.image_size,
    use_system_prompt=args.use_system_prompt,
    system_prompt=args.system_prompt,
    bot_task=args.bot_task,
    diff_infer_steps=args.diff_infer_steps,
    verbose=args.verbose,
    max_new_tokens=args.max_new_tokens,
    guidance_scale=args.guidance_scale,
    image=image_input,
    infer_align_image_size=args.infer_align_image_size,
    debug_npu=args.debug_npu,
    use_taylor_cache=args.use_taylor_cache,
    taylor_cache_interval=args.taylor_cache_interval,
    taylor_cache_order=args.taylor_cache_order,
    taylor_cache_enable_first_enhance=args.taylor_cache_enable_first_enhance,
    taylor_cache_first_enhance_steps=args.taylor_cache_first_enhance_steps,
    taylor_cache_enable_tailing_enhance=args.taylor_cache_enable_tailing_enhance,
    taylor_cache_tailing_enhance_steps=args.taylor_cache_tailing_enhance_steps,
    taylor_cache_low_freqs_order=args.taylor_cache_low_freqs_order,
    taylor_cache_high_freqs_order=args.taylor_cache_high_freqs_order,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_run_image_gen_npu.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run_image_gen_npu.py tests/test_run_image_gen_npu.py
git commit -m "feat: thread guidance scale into npu runner"
```

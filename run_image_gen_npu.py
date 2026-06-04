# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 on NPU")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Image to run. For multiple images, use comma-separated paths "
            "(e.g., 'img1.png,img2.png,img3.png')"
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="Attention implementation. Keep 'sdpa' for NPU unless you have validated another backend.",
    )
    parser.add_argument(
        "--moe-impl",
        type=str,
        default="eager",
        choices=["eager", "flashinfer"],
        help="MoE implementation. Keep 'eager' for NPU unless you have a validated custom backend.",
    )
    parser.add_argument("--torch-dtype", type=str, default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained. Use 'auto' by default or 'none' to disable it.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument(
        "--image-size",
        type=str,
        default="auto",
        help=(
            "'auto' means image size is determined by the model. Alternatively, it can be in the "
            "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes."
        ),
    )
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        help=(
            "Use system prompt. 'None' means no system prompt; 'dynamic' means "
            "the system prompt is determined by --bot-task; 'en_vanilla', "
            "'en_recaption', 'en_think_recaption' and 'en_unified' are four "
            "predefined system prompts; 'custom' means using the custom system "
            "prompt. When using 'custom', --system-prompt must be provided. "
            "Default to load from the model generation config."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt. Used when --use-system-prompt is 'custom'.",
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        choices=["image", "auto", "recaption", "think_recaption"],
        help=(
            "Type of task for the model. 'image' for direct image generation; "
            "'auto' for text generation; 'recaption' for re-write->image; "
            "'think_recaption' for think->re-write->image. "
            "Default to load from the model generation config."
        ),
    )
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose level")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override the diffusion guidance scale used during image generation. Use 1.0 to disable extra guidance.",
    )
    parser.add_argument("--rewrite", type=int, default=0, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        help="Whether to align the target image size to the src image size.",
    )
    parser.add_argument(
        "--debug-npu",
        action="store_true",
        help="Print NPU device, dtype, device_map, and condition-image tensor diagnostics.",
    )
    parser.add_argument("--use-taylor-cache", action="store_true", help="Use Taylor Cache when sampling.")
    parser.add_argument("--taylor-cache-interval", type=int, default=5, help="Interval of Taylor Cache.")
    parser.add_argument("--taylor-cache-order", type=int, default=2, help="Order of Taylor Cache.")
    parser.add_argument(
        "--taylor-cache-enable-first-enhance",
        action="store_true",
        help="Enable first enhance when using Taylor Cache.",
    )
    parser.add_argument(
        "--taylor-cache-first-enhance-steps",
        type=int,
        default=3,
        help="First enhance steps when using Taylor Cache (>2).",
    )
    parser.add_argument(
        "--taylor-cache-enable-tailing-enhance",
        action="store_true",
        help="Enable tailing enhance when using Taylor Cache.",
    )
    parser.add_argument(
        "--taylor-cache-tailing-enhance-steps",
        type=int,
        default=1,
        help="Tailing enhance steps when using Taylor Cache.",
    )
    parser.add_argument(
        "--taylor-cache-low-freqs-order",
        type=int,
        default=2,
        help="Low freqs order when using Taylor Cache.",
    )
    parser.add_argument(
        "--taylor-cache-high-freqs-order",
        type=int,
        default=2,
        help="High freqs order when using Taylor Cache.",
    )
    return parser.parse_args()


def parse_image_input(image_arg):
    if not image_arg:
        return None

    image_paths = [path.strip() for path in image_arg.split(",")]
    image_paths = [path for path in image_paths if path]
    if not image_paths:
        return None
    if len(image_paths) == 1:
        return image_paths[0]
    return image_paths


def build_model_kwargs(args):
    device_map = None if args.device_map == "none" else args.device_map
    return {
        "attn_implementation": args.attn_impl,
        "torch_dtype": args.torch_dtype,
        "device_map": device_map,
        "moe_impl": args.moe_impl,
        "moe_drop_tokens": True,
    }


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import random

    import numpy as np
    import torch

    if enable:
        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed_all(global_seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
        torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def ensure_npu_runtime():
    try:
        import torch
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is required for run_image_gen_npu.py. Install the Ascend PyTorch stack first."
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available. Check your Ascend environment and set_env.sh.")


def rewrite_prompt_if_needed(args):
    if not args.rewrite:
        return args.prompt

    from PE.deepseek import DeepSeekClient
    from PE.system_prompt import system_prompt_text_rendering, system_prompt_universal

    deepseek_key_id = os.getenv("DEEPSEEK_KEY_ID")
    deepseek_key_secret = os.getenv("DEEPSEEK_KEY_SECRET")
    if not deepseek_key_id or not deepseek_key_secret:
        raise ValueError(
            "DeepSeek API key is not set. Set DEEPSEEK_KEY_ID and DEEPSEEK_KEY_SECRET before using --rewrite."
        )

    deepseek_client = DeepSeekClient(deepseek_key_id, deepseek_key_secret)
    system_prompt = system_prompt_universal
    if getattr(args, "sys_deepseek_prompt", None) == "text_rendering":
        system_prompt = system_prompt_text_rendering
    prompt, _ = deepseek_client.run_single_recaption(system_prompt, args.prompt)
    print(f"rewrite prompt: {prompt}")
    return prompt


def main(args):
    ensure_npu_runtime()

    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    from hunyuan_image_3 import HunyuanImage3ForCausalMM

    print(
        "Using the NPU entrypoint only. Core Hunyuan modules still contain CUDA-specific code paths "
        "and may require additional patching to run end-to-end on Ascend."
    )

    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **build_model_kwargs(args))
    model.load_tokenizer(args.model_id)
    if args.debug_npu:
        print(f"NPU debug: model.device={model.device}, dtype={model.dtype}")
        print(f"NPU debug: hf_device_map={getattr(model, 'hf_device_map', None)}")

    image_input = parse_image_input(args.image)
    prompt = rewrite_prompt_if_needed(args)

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
        debug_npu=args.debug_npu,
        infer_align_image_size=args.infer_align_image_size,
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
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    samples[0].save(args.save)
    print(f"Image saved to {args.save}")
    return cot_text


if __name__ == "__main__":
    main(parse_args())

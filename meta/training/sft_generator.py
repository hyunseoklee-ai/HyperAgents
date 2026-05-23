"""
Phase 1 — Generator warm-start (LoRA SFT on self-bootstrapped trajectories).

Reads the JSONL produced by `data_prep.py` and trains a single LoRA adapter on
top of Qwen3.5-4B. Uses TRL's SFTTrainer with `assistant_only_loss=True` so
prompts (system + user / tool-result turns) are masked.

After this script finishes you have a checkpoint at
`./meta/training/checkpoints/phase1_v0/`. Load it into vLLM via
`--enable-lora --lora-modules qwen3.5-4b-phase1=./meta/training/checkpoints/phase1_v0`
for the next round of evaluation.

Usage:
  python3.11 -m meta.training.sft_generator --config meta/training/lora_config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_model_class():
    """Return AutoModelForCausalLM. Liger integration happens through TRL's
    SFTConfig(use_liger_kernel=True), which patches the loaded model in-place
    to use liger's fused linear+CE without breaking TRL's logits-slicing path."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   type=Path, default=Path("meta/training/lora_config.yaml"))
    ap.add_argument("--data",     type=Path, default=None,
                    help="Override the JSONL path from the config")
    ap.add_argument("--output",   type=Path, default=None,
                    help="Override training.output_dir from the config")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap on steps for smoke runs; -1 means train through epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_path = args.data   or Path(cfg["data"]["output_jsonl"])
    out_dir   = args.output or Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data ----------------------------------------------------------
    from datasets import load_dataset
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python3.11 -m meta.training.data_prep --config {args.config}` first.")
    print(f"Loading SFT data from {data_path}")
    ds = load_dataset("json", data_files=str(data_path), split="train")
    print(f"  {len(ds)} examples")

    # ---- 2. Model + tokenizer --------------------------------------------
    import torch
    from transformers import AutoTokenizer

    base = cfg["model"]["base_id"]
    print(f"Loading tokenizer: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # TRL's assistant_only_loss requires `{% generation %}` blocks around the
    # assistant content. Qwen3.5's default template lacks them; override with a
    # minimal ChatML-compatible template that wraps assistant content properly.
    # Reasoning content (if present) is included in the generation block too.
    _TRAINING_CHAT_TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|im_start|>system\n{{ message.content }}<|im_end|>\n"
        "{%- elif message.role == 'user' %}"
        "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
        "{%- elif message.role == 'assistant' %}"
        "<|im_start|>assistant\n{% generation %}{{ message.content }}{% endgeneration %}<|im_end|>\n"
        "{%- elif message.role == 'tool' %}"
        "<|im_start|>tool\n{{ message.content }}<|im_end|>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
    )
    tok.chat_template = _TRAINING_CHAT_TEMPLATE
    print("[tokenizer] applied training-compatible chat template ({% generation %} markers)")

    Loader = _load_model_class()
    print(f"Loading base model: {base} (bf16, {cfg['model']['attn_implementation']} attn)")
    model = Loader.from_pretrained(
        base,
        torch_dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"],
        trust_remote_code=True,
    )

    # ---- 3. LoRA ----------------------------------------------------------
    from peft import LoraConfig, get_peft_model
    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=lc["task_type"],
        target_modules=lc["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ---- 4. SFTTrainer ----------------------------------------------------
    from trl import SFTTrainer, SFTConfig
    tr = cfg["training"]
    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        logging_steps=tr["logging_steps"],
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        bf16=tr["bf16"],
        optim=tr["optim"],
        packing=tr["packing"],
        assistant_only_loss=tr["assistant_only_loss"],
        max_length=tr["max_length"],
        max_steps=args.max_steps,
        report_to=["none"],
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )
    # TRL-integrated liger (fused linear+CE) — removes the (vocab × seq_len) logits
    # materialization that caused OOM with Qwen3.5-4B at 32K seq.
    if tr.get("use_liger_kernel", False):
        sft_kwargs["use_liger_kernel"] = True
        if "liger_kernel_config" in tr:
            sft_kwargs["liger_kernel_config"] = tr["liger_kernel_config"]
        print("[trl] use_liger_kernel=True (fused linear+CE; long-seq OOM mitigation)")
    sft_cfg = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    print(f"\nTraining: {len(ds)} examples × {tr['num_train_epochs']} epochs"
          f"  -> output {out_dir}")
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"\nLoRA saved to {out_dir}")
    print(f"To serve via vLLM:")
    print(f"  python3.11 -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {base} --enable-lora \\")
    print(f"      --lora-modules qwen3.5-4b-phase1={out_dir} \\")
    print(f"      --port 8001 --max-model-len 131072")


if __name__ == "__main__":
    main()

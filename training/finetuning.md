# KugelAudio Fine-Tuning Implementation Plan

## Table of Contents

- [Overview](#overview)
- [Information Availability Assessment](#information-availability-assessment)
- [Architecture Recap](#architecture-recap)
- [Fine-Tuning Strategy](#fine-tuning-strategy)
- [Phase 1: Data Preparation](#phase-1-data-preparation)
- [Phase 2: Training Infrastructure](#phase-2-training-infrastructure)
- [Phase 3: Training Loop Implementation](#phase-3-training-loop-implementation)
- [Phase 4: Evaluation & Export](#phase-4-evaluation--export)
- [Hardware Requirements](#hardware-requirements)
- [Appendix: Key Code References](#appendix-key-code-references)

---

## Overview

This document provides a detailed implementation plan for fine-tuning the KugelAudio TTS model. KugelAudio is built on the [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) architecture — a hybrid **Autoregressive (AR) + Diffusion** text-to-speech system. The model uses a Qwen2-based LLM backbone for text understanding and an acoustic VAE + diffusion head for speech generation.

---

## Information Availability Assessment

### What is available in this repository (kugelaudio-open)

| Component | Status | Location |
|-----------|--------|----------|
| **Full model architecture** (training variant) | ✅ Available | `src/kugelaudio_open/models/kugelaudio_model.py` — `KugelAudioForConditionalGeneration` |
| **Forward pass with loss computation** | ✅ Available | `KugelAudioForConditionalGeneration.forward()` — includes both CE loss hooks and diffusion loss |
| **Diffusion head** | ✅ Available | `src/kugelaudio_open/models/diffusion_head.py` — `KugelAudioDiffusionHead` |
| **Acoustic tokenizer** (encoder + decoder) | ✅ Available | `src/kugelaudio_open/models/tokenizer.py` — `KugelAudioAcousticTokenizerModel` |
| **Model configs** (7B + 1.5B) | ✅ Available | `src/kugelaudio_open/configs/kugelaudio_7b.json`, `kugelaudio_1.5b.json` |
| **Text tokenizer** | ✅ Available | Uses Qwen2.5 tokenizer via `src/kugelaudio_open/processors/text_tokenizer.py` |
| **Prompt format / template** | ✅ Available | `src/kugelaudio_open/processors/kugelaudio_processor.py` — system prompt, voice input, text input, speech output sections |
| **Special token IDs** | ✅ Available | `speech_start_id=151652`, `speech_end_id=151653`, `speech_diffusion_id=151654`, `eos_token_id=151643` |
| **Noise scheduler** | ✅ Available | `src/kugelaudio_open/schedule/dpm_solver.py` — DPM-Solver with configurable DDPM settings |
| **Speech scaling factors** | ✅ Available | `speech_scaling_factor` and `speech_bias_factor` buffers on the model |
| **Gradient checkpointing support** | ✅ Available | `supports_gradient_checkpointing = True` on `KugelAudioPreTrainedModel` |
| **Weight initialization** | ✅ Available | `_init_weights()` method handles all component types |
| **Inference model** | ✅ Available | `src/kugelaudio_open/models/kugelaudio_inference.py` — `KugelAudioForConditionalGenerationInference` |
| **Pre-encoded voice format** | ✅ Available | `.pt` files with `acoustic_mean` tensor, registry in `voices/voices.json` |
| **Audio processing utilities** | ✅ Available | `src/kugelaudio_open/processors/audio_processor.py` |
| **Training script** | ❌ Not included | Must be written (this plan) |
| **Data preprocessing pipeline** | ❌ Not included | Must be written (this plan) |
| **Dataset** | ❌ Not included | Original training used ~200K hours from YODAS2 |

### What is available in Microsoft VibeVoice

| Component | Status | Notes |
|-----------|--------|-------|
| **ASR fine-tuning code (LoRA)** | ✅ Available | `finetuning-asr/` — uses HuggingFace Trainer + PEFT, good reference pattern |
| **ASR data format** | ✅ Available | JSON labels with segments (speaker, text, start, end) |
| **TTS training code** | ❌ Removed | Removed due to misuse concerns (Sept 2025) |
| **Architecture documentation** | ✅ Available | TTS report: [arxiv.org/pdf/2508.19205](https://arxiv.org/pdf/2508.19205) |
| **Checkpoint conversion script** | ✅ Available | `vibevoice/scripts/convert_nnscaler_checkpoint_to_transformers.py` |

### Summary

**All required architectural information for fine-tuning is available.** The `KugelAudioForConditionalGeneration` class already implements a complete training-ready forward pass with both autoregressive and diffusion losses. The missing pieces are: (1) a training script/loop, (2) a data preprocessing pipeline, and (3) dataset preparation utilities. These are detailed in the plan below.

---

## Architecture Recap

```
┌─────────────────────────────────────────────────────────────┐
│                    KugelAudio Architecture                   │
│                                                             │
│  ┌──────────────┐    ┌─────────────────────────────────┐    │
│  │ Text Input   │───▸│ Qwen2 LLM Backbone (28 layers)  │    │
│  │ (tokenized)  │    │ hidden_size: 3584 (7B) / 1536   │    │
│  └──────────────┘    │           (1.5B)                 │    │
│                      └────────────┬────────────────────┘    │
│                                   │                          │
│  ┌──────────────┐                 │                          │
│  │ Voice Embed  │                 │                          │
│  │ (.pt files)  │───▸ acoustic ───┘                          │
│  └──────────────┘    connector                               │
│                                                             │
│              ┌────────────────────┴───────────────────┐      │
│              │                                        │      │
│              ▼                                        ▼      │
│  ┌────────────────────┐              ┌──────────────────┐   │
│  │ LM Head (CE loss)  │              │ Diffusion Head   │   │
│  │ Next-token predict  │              │ (diffusion loss) │   │
│  │ for speech tokens   │              │ DDPM v-prediction│   │
│  └────────────────────┘              └────────┬─────────┘   │
│                                               │              │
│                                               ▼              │
│                                    ┌────────────────────┐   │
│                                    │ Acoustic Decoder   │   │
│                                    │ (VAE latent→audio) │   │
│                                    └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Two Loss Functions (Already Implemented)

1. **Cross-Entropy (CE) Loss**: Standard next-token prediction on the speech token sequence (computed externally in the training loop using `logits` and `labels`).
2. **Diffusion Loss**: MSE loss between predicted and target noise/velocity in the diffusion process. Computed inside `KugelAudioForConditionalGeneration.forward()` and returned as `diffusion_loss`.

### Key Model Parameters (7B variant)

| Parameter | Value |
|-----------|-------|
| LLM hidden size | 3584 |
| LLM layers | 28 |
| Attention heads | 28 (4 KV heads, GQA) |
| Acoustic VAE dim | 64 |
| Diffusion head layers | 4 |
| DDPM steps (training) | 1000 |
| DDPM batch multiplier | 4 |
| Prediction type | v_prediction |
| Beta schedule | cosine |
| Speech compression ratio | 3200 (7.5 Hz frame rate) |

---

## Fine-Tuning Strategy

### Recommended Approach: LoRA Fine-Tuning

For most fine-tuning scenarios, **LoRA (Low-Rank Adaptation)** is recommended because:

- The full model is 7B parameters (~19 GB VRAM for inference alone)
- Full fine-tuning requires 4–8× more memory for optimizer states and gradients
- LoRA keeps the base model frozen and trains small adapter matrices
- Aligns with the approach used in VibeVoice ASR fine-tuning

### Target Modules for LoRA

Based on the architecture, the following modules should be targeted:

```python
# LLM backbone (Qwen2 attention + MLP)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP (SwiGLU)
]
```

### Components to Train

| Component | LoRA Fine-Tune | Full Fine-Tune | Notes |
|-----------|:--------------:|:--------------:|-------|
| LLM backbone (Qwen2) | LoRA adapters | All weights | Main adaptation target |
| LM head | Frozen | Trainable | Tied to embeddings in 1.5B variant |
| Diffusion head | Full (small) | Full | Only 4 layers, small parameter count |
| Acoustic connector | Full (small) | Full | 2 linear layers, bridges VAE → LLM |
| Acoustic tokenizer | Frozen | Frozen | Keep frozen — audio codec should not change |

### Alternative: Full Fine-Tuning

For maximum quality (e.g., adapting to a new language family), full fine-tuning of the LLM backbone + diffusion head can be done with:

- DeepSpeed ZeRO Stage 2/3 or FSDP
- Gradient checkpointing enabled
- BF16 mixed precision
- Minimum 4× H100 80GB GPUs

---

## Phase 1: Data Preparation

### 1.1 Dataset Format

Each training sample consists of a paired text transcript and audio file. The recommended format:

```
dataset/
├── manifest.jsonl          # One JSON object per line
├── audio/
│   ├── sample_0001.wav
│   ├── sample_0002.wav
│   └── ...
```

**manifest.jsonl** format:

```json
{
  "audio_path": "audio/sample_0001.wav",
  "text": "Speaker 0: Hello, this is a sample sentence.",
  "speaker_id": 0,
  "language": "de",
  "duration_seconds": 4.2
}
```

### 1.2 Audio Preprocessing Pipeline

```python
# Preprocessing steps (to be implemented in training/data_preprocessing.py)

# 1. Resample all audio to 24 kHz mono
# 2. Normalize audio to -25 dB FS (matching AudioProcessor defaults)
# 3. Filter out samples shorter than 1 second or longer than 30 seconds
# 4. Filter out samples with low SNR or corrupted audio
# 5. Encode audio through acoustic tokenizer encoder to get VAE latents
```

Key parameters from the model config:

- **Sample rate**: 24,000 Hz
- **Channels**: 1 (mono)
- **Target dB FS**: -25
- **Speech compression ratio**: 3200 (24000 / 3200 = 7.5 frames/sec)
- **VAE latent dimension**: 64

### 1.3 Training Sample Construction

Each training sample must be formatted to match the prompt template used during the original training. Based on the processor code (`kugelaudio_processor.py`):

```
<system_prompt> Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.
 Voice input:
 Speaker 0: [VAE_TOKEN × N_voice_frames]
 Text input:
 Speaker 0: {text}
 Speech output:
<speech_start>[diffusion_token × N_target_frames]<speech_end>
```

Where:
- `[VAE_TOKEN]` = token ID `151654` (placeholder for voice embedding injection)
- `<speech_start>` = token ID `151652`
- `<speech_end>` = token ID `151653`
- Voice frames come from a reference audio (3–10 seconds recommended)
- Target frames come from the target audio to synthesize

### 1.4 Data Collator

A custom data collator is needed to handle:

1. **Padding** text sequences to uniform length within a batch
2. **Constructing masks**:
   - `attention_mask`: standard padding mask
   - `acoustic_input_mask`: positions where VAE embeddings replace text embeddings (voice input section)
   - `acoustic_loss_mask`: positions where diffusion loss should be computed (speech output section)
   - `speech_masks`: which frames in the speech tensor are valid
   - `speeches_loss_input`: which speech frames participate in diffusion loss
3. **Labels**: shifted input_ids for CE loss (with `-100` at non-speech positions if only training on speech tokens)
4. **Speech tensors**: the raw audio or pre-computed VAE latents for the target speech

---

## Phase 2: Training Infrastructure

### 2.1 Dependencies

Add to `pyproject.toml` under a new `[project.optional-dependencies]` group:

```toml
[project.optional-dependencies]
training = [
    "peft>=0.7.0",           # LoRA adapters
    "deepspeed>=0.12.0",     # Distributed training (ZeRO)
    "wandb>=0.16.0",         # Experiment tracking
    "datasets>=2.14.0",      # HuggingFace datasets
    "bitsandbytes>=0.41.0",  # 8-bit optimizers (optional)
]
```

### 2.2 Project Structure

```
training/
├── finetuning.md                # This document
├── train.py                     # Main training script
├── data_preprocessing.py        # Audio preprocessing and VAE encoding
├── dataset.py                   # PyTorch Dataset + DataLoader
├── collator.py                  # Custom data collator for batching
├── configs/
│   ├── lora_7b.yaml             # LoRA config for 7B model
│   ├── lora_1.5b.yaml           # LoRA config for 1.5B model
│   └── full_finetune.yaml       # Full fine-tune config
└── scripts/
    ├── preprocess_data.sh       # Data preprocessing script
    ├── run_training.sh          # Training launch script
    └── export_model.sh          # Model export / merge script
```

### 2.3 Configuration Files

**LoRA config (`configs/lora_7b.yaml`)**:

```yaml
# Model
model_path: "kugelaudio/kugelaudio-0-open"
model_variant: "7b"

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
# These small modules are fully trainable (not LoRA):
trainable_modules:
  - "prediction_head"        # Diffusion head (~small)
  - "acoustic_connector"     # Speech connector (~small)

# Data
data_dir: "./data"
max_audio_duration: 30.0     # seconds
min_audio_duration: 1.0      # seconds
speech_compression_ratio: 3200

# Training
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true

# Diffusion
ddpm_batch_mul: 4            # From model config
diffusion_loss_weight: 1.0   # Weight for diffusion loss vs CE loss

# Logging
logging_steps: 10
save_steps: 500
eval_steps: 500
report_to: "wandb"
```

---

## Phase 3: Training Loop Implementation

### 3.1 Model Loading

```python
import torch
from peft import LoraConfig, get_peft_model
from kugelaudio_open.models.kugelaudio_model import KugelAudioForConditionalGeneration

# Load the training-variant model (NOT the inference-only variant)
model = KugelAudioForConditionalGeneration.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    torch_dtype=torch.bfloat16,
)

# Apply LoRA to the LLM backbone
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Ensure diffusion head and acoustic connector are trainable
for name, param in model.named_parameters():
    if "prediction_head" in name or "acoustic_connector" in name:
        param.requires_grad = True

# Freeze acoustic tokenizer (encoder + decoder)
for name, param in model.named_parameters():
    if "acoustic_tokenizer" in name:
        param.requires_grad = False

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### 3.2 Dataset Class

```python
import json
import torch
from torch.utils.data import Dataset

class KugelAudioTTSDataset(Dataset):
    """Dataset for KugelAudio TTS fine-tuning.

    Each sample contains:
    - input_ids: tokenized prompt (system + voice ref + text + speech tokens)
    - labels: shifted input_ids for CE loss
    - speech_tensors: VAE-encoded acoustic features of the target audio
    - speech_masks: valid frame mask for the speech tensors
    - acoustic_input_mask: positions where voice embeddings are injected
    - acoustic_loss_mask: positions where diffusion loss is computed
    - speeches_loss_input: subset of speech_masks for diffusion training
    """

    def __init__(self, manifest_path, tokenizer, acoustic_tokenizer,
                 audio_processor, speech_compression_ratio=3200,
                 max_duration=30.0, min_duration=1.0):
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                dur = sample.get("duration_seconds", 0)
                if min_duration <= dur <= max_duration:
                    self.samples.append(sample)

        self.tokenizer = tokenizer
        self.acoustic_tokenizer = acoustic_tokenizer
        self.audio_processor = audio_processor
        self.speech_compression_ratio = speech_compression_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load and preprocess audio
        audio = self.audio_processor.load_audio(sample["audio_path"])

        # 2. Encode audio through acoustic tokenizer to get VAE latents
        with torch.no_grad():
            # audio shape: [samples] -> [1, 1, samples]
            frames = self.acoustic_tokenizer.encode(
                audio.unsqueeze(0).unsqueeze(0)
            )
            # Extract VAE mean (latent representation)
            vae_latents = frames[0][0].sample("gaussian")[0]
            # vae_latents shape: [1, num_frames, 64]

        # 3. Build token sequence matching the training template
        # (See Section 1.3 for the full template)
        # ... tokenization and mask construction ...

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speech_tensors": vae_latents,
            "speech_masks": speech_masks,
            "acoustic_input_mask": acoustic_input_mask,
            "acoustic_loss_mask": acoustic_loss_mask,
            "speeches_loss_input": speeches_loss_input,
        }
```

### 3.3 Training Step

The core training step leverages the existing `forward()` method:

```python
def training_step(model, batch, diffusion_loss_weight=1.0):
    """Single training step with combined CE + diffusion loss."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        speech_tensors=batch["speech_tensors"],
        speech_masks=batch["speech_masks"],
        acoustic_input_mask=batch["acoustic_input_mask"],
        acoustic_loss_mask=batch["acoustic_loss_mask"],
        speeches_loss_input=batch["speeches_loss_input"],
        ddpm_batch_mul=4,  # From config: ddpm_batch_mul
        speech_type="vae",  # Use pre-encoded VAE latents
    )

    # 1. CE loss: computed on logits vs labels (speech token positions only)
    logits = outputs.logits
    labels = batch["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # 2. Diffusion loss: already computed in forward()
    diffusion_loss = outputs.diffusion_loss

    # 3. Combined loss
    total_loss = ce_loss + diffusion_loss_weight * diffusion_loss

    return total_loss, ce_loss, diffusion_loss
```

### 3.4 Training Script Outline (`train.py`)

```python
"""Main training script for KugelAudio TTS fine-tuning."""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from kugelaudio_open.models.kugelaudio_model import KugelAudioForConditionalGeneration
from kugelaudio_open.processors.text_tokenizer import KugelAudioTextTokenizer
from kugelaudio_open.processors.audio_processor import AudioProcessor

# from training.dataset import KugelAudioTTSDataset
# from training.collator import KugelAudioCollator


def main(args):
    # 1. Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 2. Load model (training variant)
    model = KugelAudioForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    # 3. Apply LoRA (if enabled)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Keep small modules fully trainable
        for name, param in model.named_parameters():
            if "prediction_head" in name or "acoustic_connector" in name:
                param.requires_grad = True
            if "acoustic_tokenizer" in name:
                param.requires_grad = False

    # 4. Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 5. Create dataset and dataloader
    # dataset = KugelAudioTTSDataset(...)
    # collator = KugelAudioCollator(...)
    # dataloader = DataLoader(dataset, collate_fn=collator, ...)

    # 6. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # scheduler = get_cosine_schedule_with_warmup(...)

    # 7. Prepare with accelerator
    # model, optimizer, dataloader, scheduler = accelerator.prepare(
    #     model, optimizer, dataloader, scheduler
    # )

    # 8. Training loop
    # for epoch in range(args.num_train_epochs):
    #     for step, batch in enumerate(dataloader):
    #         with accelerator.accumulate(model):
    #             total_loss, ce_loss, diff_loss = training_step(
    #                 model, batch, args.diffusion_loss_weight
    #             )
    #             accelerator.backward(total_loss)
    #             accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()

    # 9. Save
    # accelerator.save_state(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()
    main(args)
```

---

## Phase 4: Evaluation & Export

### 4.1 Evaluation Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| **Training loss** | CE + diffusion loss curves | wandb / tensorboard |
| **MOS (Mean Opinion Score)** | Human evaluation of naturalness | Manual A/B testing |
| **Speaker similarity** | Cosine similarity of speaker embeddings | WavLM / Resemblyzer |
| **Word Error Rate (WER)** | Intelligibility via ASR transcription | Whisper |
| **RTF (Real-Time Factor)** | Generation speed benchmark | Timed inference |

### 4.2 Inference after Fine-Tuning

**With LoRA adapters** (no merge):

```python
from peft import PeftModel
from kugelaudio_open.models.kugelaudio_inference import (
    KugelAudioForConditionalGenerationInference,
)

# Load base inference model
base_model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load LoRA adapters on top
model = PeftModel.from_pretrained(base_model, "./output/checkpoint-best")
model.eval()
model.model.strip_encoders()

# Generate as usual
# processor = KugelAudioProcessor.from_pretrained(...)
# inputs = processor(text="...", voice="default", return_tensors="pt")
# outputs = model.generate(**inputs)
```

**Merge LoRA into base model** (for deployment):

```python
from peft import PeftModel

# Load training model + LoRA
model = KugelAudioForConditionalGeneration.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, "./output/checkpoint-best")

# Merge and save
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

### 4.3 Voice Embedding Export

To create new pre-encoded voice embeddings from fine-tuned model:

```python
# Encode a reference audio clip (3-10 seconds) through the acoustic tokenizer
audio = audio_processor.load_audio("reference_voice.wav")
with torch.no_grad():
    frames = model.model.acoustic_tokenizer.encode(
        audio.unsqueeze(0).unsqueeze(0).to(device)
    )
    acoustic_mean = frames[0][0].mean  # Extract the mean (not sampled)

# Save as voice cache
voice_cache = {"acoustic_mean": acoustic_mean.cpu()}
torch.save(voice_cache, "voices/new_voice.pt")

# Register in voices.json
# { "new_voice": { "file": "new_voice.pt", "description": "...", "language": "..." } }
```

> **Note**: Voice encoding requires the acoustic tokenizer **encoder**, which is stripped during inference. Use the training model variant (`KugelAudioForConditionalGeneration`) for voice encoding.

---

## Hardware Requirements

### LoRA Fine-Tuning (Recommended)

| Model | GPUs | VRAM per GPU | Effective Batch Size | Estimated Time |
|-------|------|-------------|---------------------|----------------|
| 7B | 1× A100 80GB | ~40 GB | 1 × 8 grad_accum = 8 | ~2–4 hours per epoch (1K samples) |
| 7B | 4× A100 80GB | ~40 GB | 4 × 2 grad_accum = 8 | ~0.5–1 hour per epoch (1K samples) |
| 1.5B | 1× A100 40GB | ~20 GB | 1 × 8 grad_accum = 8 | ~1–2 hours per epoch (1K samples) |
| 1.5B | 1× RTX 4090 24GB | ~20 GB | 1 × 8 grad_accum = 8 | ~2–3 hours per epoch (1K samples) |

### Full Fine-Tuning

| Model | GPUs | VRAM per GPU | Notes |
|-------|------|-------------|-------|
| 7B | 4× H100 80GB | ~70 GB | DeepSpeed ZeRO-3 + gradient checkpointing |
| 7B | 8× A100 80GB | ~60 GB | DeepSpeed ZeRO-2 + gradient checkpointing |
| 1.5B | 2× A100 80GB | ~50 GB | DeepSpeed ZeRO-2 |

### Key Training Settings

```bash
# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_ALLOW_TF32=1
export NCCL_P2P_DISABLE=0
```

---

## Appendix: Key Code References

### Existing Code That Supports Fine-Tuning

| File | Class/Function | Relevance |
|------|---------------|-----------|
| `src/kugelaudio_open/models/kugelaudio_model.py` | `KugelAudioForConditionalGeneration` | Training model with `forward()` that computes CE + diffusion loss |
| `src/kugelaudio_open/models/kugelaudio_model.py` | `forward_speech_features()` | Processes speech through acoustic tokenizer + connector |
| `src/kugelaudio_open/models/kugelaudio_model.py` | `KugelAudioModel.strip_encoders()` | Removes encoder (for inference only — do NOT call during training) |
| `src/kugelaudio_open/models/diffusion_head.py` | `KugelAudioDiffusionHead` | Small diffusion network, should be fully trainable |
| `src/kugelaudio_open/models/tokenizer.py` | `KugelAudioAcousticTokenizerModel` | Acoustic VAE — encoder (training) + decoder (inference) |
| `src/kugelaudio_open/configs/model_config.py` | `KugelAudioConfig` | Full model config with all sub-configs |
| `src/kugelaudio_open/configs/kugelaudio_7b.json` | — | 7B model hyperparameters |
| `src/kugelaudio_open/configs/kugelaudio_1.5b.json` | — | 1.5B model hyperparameters |
| `src/kugelaudio_open/processors/kugelaudio_processor.py` | `KugelAudioProcessor.__call__()` | Prompt template construction (training format) |
| `src/kugelaudio_open/processors/audio_processor.py` | `AudioProcessor` | Audio loading, resampling, normalization |
| `src/kugelaudio_open/schedule/dpm_solver.py` | `DPMSolverMultistepScheduler` | Noise scheduler with `add_noise()` and `get_velocity()` |

### VibeVoice References

| Resource | URL | Relevance |
|----------|-----|-----------|
| VibeVoice ASR LoRA fine-tuning | [github.com/microsoft/VibeVoice/finetuning-asr](https://github.com/microsoft/VibeVoice/tree/main/finetuning-asr) | Reference pattern for LoRA fine-tuning with HF Trainer |
| VibeVoice TTS technical report | [arxiv.org/pdf/2508.19205](https://arxiv.org/pdf/2508.19205) | Architecture details, training methodology |
| YODAS2 dataset | [huggingface.co/datasets/espnet/yodas](https://huggingface.co/datasets/espnet/yodas) | Original training data source |
| Qwen2.5 tokenizer | [huggingface.co/Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Text tokenizer used by the model |

### Key Special Token IDs

| Token | ID | Purpose |
|-------|-----|---------|
| `speech_start` | 151652 | Marks beginning of speech output |
| `speech_end` | 151653 | Marks end of speech output |
| `speech_diffusion` | 151654 | Placeholder for diffusion-generated speech frames |
| `eos` | 151643 | End of sequence |

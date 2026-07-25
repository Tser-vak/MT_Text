"""Day-6: ONE place MedGemma gets constructed from. Every consumer (train.py,
evaluate_model.py) loads through this module instead of re-typing MODEL_ID or
the quantization/LoRA wiring -- today those are duplicated in
measure_lengths.py and the notebook; this module is the new single source
going forward (those two are left alone, see the plan's do-not-touch list).

CRITICAL: no `bitsandbytes` import anywhere in this file at module scope --
the actual 4-bit load call lives inside `load_base_model`, so this module
still IMPORTS cleanly on a CPU box with no CUDA/bitsandbytes installed. Only
CALLING `load_base_model`/`load_for_inference` needs the GPU box.

Read before implementing:
  - HF bitsandbytes 4-bit quantization: https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
  - MedGemma model card: https://huggingface.co/google/medgemma-4b-it
  - PEFT LoraConfig (target_modules / exclude_modules regex):
    https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
  - PEFT prepare_model_for_kbit_training:
    https://huggingface.co/docs/peft/main/en/package_reference/peft_types#peft.prepare_model_for_kbit_training
"""
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import torch

MODEL_ID = "google/medgemma-4b-it"


def load_processor() -> AutoProcessor:
    """Load MedGemma's processor. Sets `padding_side="left"` -- generation-time
    batching needs left-padding so every sequence in a batch ends at the same
    position (training uses the collator's own padding, this doesn't affect it).
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return processor


def load_base_model(quantize: bool = True) -> AutoModelForImageTextToText:
    """Load the base MedGemma weights (no LoRA yet).

        `quantize=True` (the QLoRA path): build an NF4 `BitsAndBytesConfig` and
        pass it as `quantization_config` to `AutoModelForImageTextToText.from_pretrained`.
        `quantize=False`: load full precision (e.g. for a CPU-only dry run of the
        surrounding plumbing -- MedGemma 4B will not fit in RAM/VRAM unquantized
        on modest hardware, this branch is for wiring checks, not real training).

        Returns: the loaded `AutoModelForImageTextToText` model, not yet wrapped
        by `attach_lora`.
        """
    #Configuring the packaging of data and unpackaging
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True) if quantize else None

    #uploading the model weights
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID,device_map="auto",
                                                        quantization_config=bnb_config,
                                                        dtype=torch.bfloat16)
    return model

def attach_lora(model: AutoModelForImageTextToText):
    """Wrap `model` (already 4-bit loaded + `prepare_model_for_kbit_training`'d
    by the caller) with a LoRA adapter scoped to the LANGUAGE tower only.

    Prints the trainable-parameter percentage before returning (PEFT's own
    `model.print_trainable_parameters()` is fine for this).

    Returns: the `PeftModel`-wrapped model, ready for `SFTTrainer`.
    """
    # ─────────────────────────────────────────────────────────────
    # YOUR CODE — body modeling.attach_lora
    # Goal: build a LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
    #       task_type="CAUSAL_LM", target_modules=...) scoped to
    #       language_model.*'s attention/MLP projections ONLY, then
    #       get_peft_model(model, lora_config).
    # Why:  a bare leaf-name list like ["q_proj", "v_proj", ...] ALSO matches
    #       inside the SigLIP vision tower and the multi-modal projector --
    #       adapting those wastes capacity on a task that doesn't need it and
    #       is the classic MedGemma QLoRA footgun. 0% or 100% trainable
    #       printed means it's wired wrong either way.
    # Read: https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
    # Done when (VM): trainable params print ~1-2% of total, and the printed
    #       target-module list contains no vision_tower/multi_modal_projector
    #       entries.
    raise NotImplementedError("modeling.attach_lora")
    # ─────────────────────────────────────────────────────────────


def load_for_inference(adapter_path: str | None = None):
    """Load processor + base model for GENERATION (baseline when
    `adapter_path` is None, fine-tuned when it's a saved LoRA adapter dir).
    Always 4-bit (inference doesn't need full precision). Puts the model in
    eval mode. Returns `(processor, model)`."""
    processor = load_processor()
    model = load_base_model(quantize=True)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return processor, model


if __name__ == "__main__":
    # ponytail: no real model load here -- MedGemma needs the HF gate cleared
    # (see measure_lengths.py) plus a GPU for the 4-bit path, neither of which
    # this local CPU box has. This just checks the module wiring imports and
    # exposes the right names.
    assert MODEL_ID == "google/medgemma-4b-it"
    for fn in (load_processor, load_base_model, attach_lora, load_for_inference):
        assert callable(fn)
    print("modeling.py wiring OK (load_processor/load_base_model/load_for_inference "
          "need the MedGemma HF gate + a GPU -- run those on the VM)")

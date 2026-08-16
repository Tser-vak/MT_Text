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
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import torch

MODEL_ID = "google/medgemma-4b-it"

# Anchored on the `language_model.` PATH SEGMENT, not on leaf names: peft matches
# a name LIST by suffix, and the SigLIP vision tower's attention uses those same
# leaf names (verified: SiglipAttention -> q_proj/k_proj/v_proj/out_proj), so a
# bare ["q_proj", ...] silently adapts an image encoder this project never uses.
# Passing a STRING is what makes peft switch to re.fullmatch over the full module
# path -- as a list the anchor would be ignored. Same shape as peft's own gemma4
# default, widened from q/v to attention + MLP.
LANGUAGE_TOWER_PROJECTIONS = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


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
    # attn_implementation="eager": Gemma-3's card recommends it for TRAINING --
    # the sdpa path interacts badly with attention soft-capping.
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID,device_map="auto",
                                                        quantization_config=bnb_config,
                                                        dtype=torch.bfloat16,
                                                        attn_implementation="eager")
    return model

def attach_lora(model: AutoModelForImageTextToText):
    """Wrap `model` (already 4-bit loaded + `prepare_model_for_kbit_training`'d
        by the caller) with a LoRA adapter scoped to the LANGUAGE tower only.

        Prints the trainable-parameter percentage before returning (PEFT's own
        `model.print_trainable_parameters()` is fine for this).

        Returns: the `PeftModel`-wrapped model, ready for `SFTTrainer`.
        """
    # Plain LoRA on purpose -- no rsLoRA, no DoRA. This is the plan's starting
    # config (scale = alpha/r = 2.0, which the 2e-4 LR was chosen against); both
    # variants belong in the Day-9 one-variable-at-a-time table, measured against
    # this run as the incumbent.
    lora_config = LoraConfig(r=16,
                            lora_alpha=32,
                             bias="none",
                             lora_dropout = 0.05,
                             target_modules=LANGUAGE_TOWER_PROJECTIONS,
                             task_type="CAUSAL_LM")

    peft_model = get_peft_model(model, lora_config)

    # Day-6 gate: expect ~1-2% trainable and no vision_tower entry. 0% means the
    # regex matched nothing (real nesting differs from what it assumes);
    # ~100% would mean the scoping was ignored entirely.
    peft_model.print_trainable_parameters()
    targets = sorted(peft_model.targeted_module_names)
    print(f"LoRA targets: {len(targets)} modules, e.g. {targets[:3]}")
    assert not any("vision_tower" in t or "multi_modal_projector" in t for t in targets), \
        "LoRA leaked into the vision tower -- check LANGUAGE_TOWER_PROJECTIONS"
    return peft_model


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

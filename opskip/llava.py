import inspect
from functools import wraps
import torch

from .patch import replace_method


def setup(model, state, changes, family):
    if not hasattr(model, "prepare_inputs_labels_for_multimodal"):
        raise ValueError("Use the original LLaVA/LLaVA-NeXT model loader")
    if model.model.layers[0].self_attn.__class__.__name__.startswith("Mistral"):
        raise ValueError("The released LLaVA-NeXT policy targets llava-v1.6-vicuna-7b")
    original = model.prepare_inputs_labels_for_multimodal
    signature = inspect.signature(original)

    @wraps(original.__func__)
    def prepare(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        state.reset()
        ids = values["input_ids"]
        if values.get("images") is None or ids is None or ids.shape[1] <= 1:
            return original(*args, **kwargs)
        if values.get("labels") is not None:
            raise ValueError("Op-Skip's LLaVA adapter supports generation without labels")
        values["labels"] = torch.zeros_like(ids)
        result = list(original(*bound.args, **bound.kwargs))
        if result[4] is not None:
            mask = result[5] == -100
            if result[2] is not None:
                mask &= result[2].bool()
            state.reset(mask, allow_empty_suffix=family == "llava_next")
        result[5] = None
        return tuple(result)

    replace_method(model, "prepare_inputs_labels_for_multimodal", prepare, changes)
    decoder = model.model
    decoder_original = decoder.forward

    @wraps(decoder_original.__func__)
    def decoder_forward(self, *args, **kwargs):
        try:
            return decoder_original(*args, **kwargs)
        finally:
            state.reset()

    replace_method(decoder, "forward", decoder_forward, changes)
    if state.optimized and family == "llava_next":
       
        representatives = []
        for layer in decoder.layers:
            rotary = layer.self_attn.rotary_emb
            group = id(rotary)
            if rotary.__class__.__name__ == "LlamaRotaryEmbedding" and "forward" not in rotary.__dict__:
                for previous in representatives:
                    if (type(previous) is type(rotary)
                            and getattr(previous, "base", None) == getattr(rotary, "base", None)
                            and previous.inv_freq.shape == rotary.inv_freq.shape
                            and previous.inv_freq.dtype == rotary.inv_freq.dtype
                            and previous.inv_freq.device == rotary.inv_freq.device
                            and torch.equal(previous.inv_freq, rotary.inv_freq)):
                        group = id(previous)
                        break
                else:
                    representatives.append(rotary)
            state.rotary_groups[id(rotary)] = group
    return decoder.layers

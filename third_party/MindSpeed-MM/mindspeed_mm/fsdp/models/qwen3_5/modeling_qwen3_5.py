"""Minimal FSDP registration adapter for Transformers 5.2.0 Qwen3.5.

The competition reference archive contains the Qwen3.5 configuration and a
successful reference log, but its ``fsdp/models/qwen3_5`` directory is empty.
Keep the model implementation owned by the exact Transformers release required
by the sample and register that class with MindSpeed-MM's model hub.
"""

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from mindspeed_mm.fsdp.utils.register import model_register


@model_register.register("qwen3_5")
class MindSpeedQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 model exposed through MindSpeed-MM's FSDP model registry."""


__all__ = ["MindSpeedQwen3_5ForConditionalGeneration"]

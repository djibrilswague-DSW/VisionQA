import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model():

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    return model, processor

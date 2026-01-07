
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


class NERInferenceService:
    def __init__(self, model_path: str):
        # 1. Loading Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # 2. Setting Inference Pipeline
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.nlp = pipeline(
            "token-classification",
            tokenizer=self.tokenizer,
            model=self.model,
            aggregation_strategy="simple",
            device=device,
        )

    def infer(self, text: str):
        raw_results = self.nlp(text)

        return [
            {**res, "score": float(res["score"])}
            for res in raw_results
        ]

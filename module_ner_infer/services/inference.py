
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from konlpy.tag import Mecab


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


class MeCabInferenceService:
    def __init__(self, dic_path: str):
        print(f"DEBUG: Loading MeCab with DIC_PATH: {dic_path}")
        self.mecab = Mecab(dicpath=dic_path)

    def infer(self, text: str):
        raw_results = self.mecab.pos(text)
        print(f"DEBUG MeCab Raw: {raw_results}")
        filtered_results = []
        for (word, pos) in raw_results:
            if pos == "NNP" or pos == "NNG":
                print(f"{word}, {pos}")
                filtered_results.append({
                    'entity_group': pos,
                    'score': 1,
                    'word': word,
                })

        return filtered_results

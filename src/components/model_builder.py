from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from ..components.exceptions import ModelBuildError

def build_tokenizer(cfg):
    try:
        tok = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
        return tok
    except Exception as e:
        raise ModelBuildError(f"Tokenizer load failed: {e}") from e

def build_sequence_classifier(cfg, num_labels: int, label2id=None, id2label=None):
    try:
        model_config = AutoConfig.from_pretrained(
            cfg.model.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name,
            config=model_config
        )
        return model
    except Exception as e:
        raise ModelBuildError(f"Model build failed: {e}") from e

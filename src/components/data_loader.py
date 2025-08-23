from datasets import load_dataset, DatasetDict
from typing import Dict, Any
from ..components.exceptions import DataLoadError

def load_text_classification_datasets(cfg) -> DatasetDict:
    """
    Expects JSONL or CSV with fields:
      - text field: cfg.data.text_field
      - label field: cfg.data.label_field
    """
    try:
        train_path = cfg.data.train_file
        eval_path = cfg.data.eval_file
        ext = train_path.split(".")[-1].lower()

        if ext in ("json", "jsonl"):
            data_files = {"train": train_path, "validation": eval_path}
            ds = load_dataset("json", data_files=data_files)
        elif ext == "csv":
            data_files = {"train": train_path, "validation": eval_path}
            ds = load_dataset("csv", data_files=data_files)
        else:
            raise DataLoadError(f"Unsupported data extension: .{ext}. Use JSONL or CSV.")

        # Ensure required columns exist
        for split in ["train", "validation"]:
            cols = ds[split].column_names
            if cfg.data.text_field not in cols or cfg.data.label_field not in cols:
                raise DataLoadError(
                    f"Missing columns in {split}: need '{cfg.data.text_field}' and '{cfg.data.label_field}'"
                )
        return ds
    except Exception as e:
        raise DataLoadError(f"Failed to load datasets: {e}") from e

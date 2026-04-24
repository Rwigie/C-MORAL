import os


def save_best_adapter(model, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, save_embedding_layers=True)

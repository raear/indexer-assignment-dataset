from .config import CONFIG as cfg
from .helper import create_qrels

       
def main():
    create_qrels(cfg["train_set_path"], cfg["train_qrels_path"])
    create_qrels(cfg["val_set_path"], cfg["val_qrels_path"])
    create_qrels(cfg["test_set_path"], cfg["test_qrels_path"])


if __name__ == "__main__":
    main()
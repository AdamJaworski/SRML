from pathlib import Path


class PathManager():
    def __init__(self, model_name):
        self.root_path  = Path(rf'./models/{model_name}').resolve()
        self.out_path   = self.root_path / "output"
        self.loss_file = self.root_path  / "loss.txt"
        self.create_dirs()

    def create_dirs(self):
        self.root_path.mkdir(exist_ok=True, parents=True)
        self.out_path.mkdir(exist_ok=True, parents=True)

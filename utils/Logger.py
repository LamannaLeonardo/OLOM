import os


class Logger:

    def __init__(self):
        self.log_dir = None
        self.img_dir = None
        self.models_dir = None

    def set_log_dir(self, log_dir, create_img_dir=True, create_models_dir=True):
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = f"{log_dir}/run{len([d for d in os.listdir(log_dir) if d.startswith('run')])}"
        self.img_dir = f'{self.log_dir}/imgs'
        self.models_dir = f'{self.log_dir}/models'

        os.makedirs(self.log_dir, exist_ok=True)
        if create_img_dir:
            os.makedirs(self.img_dir, exist_ok=True)
        if create_models_dir:
            os.makedirs(self.models_dir, exist_ok=True)

    def write(self, string):
        with open(f"{self.log_dir}/log", 'a') as f:
            f.write(f"\n{string}")
        print(string)

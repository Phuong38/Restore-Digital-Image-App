import yaml

class Settings:
    """
    Handle app settings
    """
    def __init__(self, configs_file: str = 'app/assets/configs/configs.yaml'):
        """
        Load configuration from config.yaml
        :param configs_file: path to configs.yaml
        """
        # Load configuration
        self.configs_file = configs_file
        self.configs = None

        self.load_configs()

    def load_configs(self):
        """
        Load configs from configs.yaml
        :return:
        """
        with open(self.configs_file, mode="r") as f:
            configs = yaml.safe_load(f)

        self.configs = configs

    def write_configs_file(self, configs: dict = None, file_path: str = None):
        """
        Write configs to file
        :param configs: Configurations will be saved
        :param file_path: Path to configs.yaml
        :return:
        """
        if file_path is None:
            file_path = self.configs_file
        if configs is None:
            configs = self.configs
        with open(file_path, mode="w") as f:
            data = yaml.dump(configs, f)
            print(f"Saved configs to: {file_path}")

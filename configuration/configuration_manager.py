from Config import Config

class ConfigurationManager:
    instance = None  # Singleton instance

    def __new__(cls):
        if cls.instance is None: # If instance is not created, create it
            cls.instance = super(ConfigurationManager, cls).__new__(cls)
            cls.instance.config = Config()
            return cls.instance
        return cls.instance

    def get_config(self):
        return self.config

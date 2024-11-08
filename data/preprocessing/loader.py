class DataLoader:
    data_dir: str
    batch_size: int
    num_workers: int
    shuffle: bool

    def __init__(self, data_dir, batch_size, num_workers, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle


    def get_data_loader(self):
        pass


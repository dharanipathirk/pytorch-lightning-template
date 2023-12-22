import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class myDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.basic_transform = transforms.compose([])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

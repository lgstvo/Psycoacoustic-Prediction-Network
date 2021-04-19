import torch
import pandas as pd

from torch.utils.data import Dataset, Dataloader

class PaDataloader(Dataset):
    def __init__(self, data_csv, data_dir, dataset_proportion=1, task='regression', num_classes=2, normalize=True, transform=None, array=False):
        self.dataset_df = pd.read_csv(data_csv)
        self.dataset_df = self.dataset_df[self.dataset_df.PA != '[]']
        #self.dataset_df = self.dataset_df[~self.dataset_df.PA.str.contains('nan')]
        self.dataset_df = self.dataset_df.dropna()
        self.dataset_df = self.dataset_df[:int(len(self.dataset_df) * dataset_proportion)]
        self.transform = transform
        self.data_dir = data_dir
        self.array = array
        self.normalize = normalize
        self.task = task
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset_df)
    
    def get_dataframe(self):
        return self.dataset_df

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        filename = row["slice_file_name"]
        fold_folder = "/fold{}/".format(row(["fold"]))
        audio_file = row.segment
        paarray = np.array(row.PA)

        if self.normalize:
            paarray = np.minimum(paarray, 99.99)

        label = paarray

        sample = {'audio_file': wavfile_to_examples(audio_file)[0], 'label': label}
        return sample

def get_dataloaders(dataset, split_prop, batch_size, shuffle, num_workers, drop_last = True):
    dataset_size = len(dataset)
    split_sizes = list((np.array(split_prop) * dataset_size).astype(int))
    split_sizes[2] += dataset_size - np.sum(split_sizes)
    split_sizes = [x.item() for x in split_sizes]
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, split_sizes)

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    print("Training Samples (Batches): {}({}), Validation Samples (Batches): {}({}), Testing Samples (Batches): {}({})".format(len(train_dataloader)*batch_size,
                        len(val_dataloader)*batch_size, len(val_dataloader),
                        len(test_dataloader)*batch_size, len(test_dataloader)))
    return train_dataloader, val_dataloader, test_dataloader
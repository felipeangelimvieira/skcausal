import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch.utils.data import Dataset


class ArgsDataset(Dataset):
    def __init__(self, *args):
        """
        Args:
            *args: Variable length arguments, each representing a dataset element.
        """
        # Convert all args to default dtype
        args = list(map(lambda x: _force_numeric_to_default_dtype(x), args))
        # Store the arguments as a list of tuples
        self.data = list(zip(*args))

    def __len__(self):
        """Returns the number of elements in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the data at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: The tuple containing the elements at the given index.
        """
        return self.data[idx]


def _force_numeric_to_default_dtype(x):
    default_dtype = torch.get_default_dtype()
    x = torch.tensor(x)

    if x.dtype in [torch.float32, torch.float64]:
        x = x.to(default_dtype)
    return x


def create_dataloader(*args, batch_size, shuffle=True):
    """Creates a PyTorch DataLoader that returns X, t, y tuples."""

    args = list(map(lambda x: _force_numeric_to_default_dtype(x), args))

    return torch.utils.data.DataLoader(
        list(zip(*args)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def create_train_val_dataloaders_from_arrays(*args, split_ratio, batch_size):
    if split_ratio:
        split_data = train_test_split(*args, test_size=split_ratio)
        train_data = split_data[::2]
        val_data = split_data[1::2]
    else:
        train_data = args
        val_data = [None] * len(args)

    train_dataloader = create_dataloader(
        *train_data, shuffle=True, batch_size=batch_size
    )
    val_dataloader = None
    if all(v is not None for v in val_data):
        val_dataloader = create_dataloader(
            *val_data, shuffle=False, batch_size=batch_size
        )

    return train_dataloader, val_dataloader


def create_train_val_dataloaders_from_dataset(
    dataset: torch.utils.data.Dataset, split_ratio, batch_size
):
    if split_ratio:
        train_size = int((1 - split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        train_dataset = dataset
        val_dataloader = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_dataloader, val_dataloader

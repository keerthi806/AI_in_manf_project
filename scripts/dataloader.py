import pandas as pd
import torch
from PIL import Image
from scipy import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DICDataset(Dataset):
    def __init__(self, paths_df: pd.DataFrame, transform: transforms = None):
        required_columns = {"image_1", "image_2", "label_path"}
        if not required_columns.issubset(paths_df):
            raise NameError(
                "The DataFrame must have 'image_1', 'image_2', 'label_path' as column names."
            )
        self.paths_df = paths_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        row = self.paths_df.iloc[idx]
        image_1_path = row["image_1"]
        image_2_path = row["image_2"]
        label_path = row["label_path"]

        tensor_tranform = transforms.ToTensor()

        image_1 = Image.open(image_1_path)
        image_2 = Image.open(image_2_path)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        mat = io.loadmat(label_path)

        if "Disp_field_1" in mat:
            disp = mat["Disp_field_1"]
        elif "Disp_field_2" in mat:
            disp = mat["Disp_field_2"]

        disp_label = torch.from_numpy(disp.transpose(2, 0, 1)).float()

        return tensor_tranform(image_1), tensor_tranform(image_2), disp_label


def get_dataloaders(df_filtered, data_frac=1, test_size=0.2, batch_size=32, seed=42):
    dataset_paths = df_filtered.sample(frac=data_frac, random_state=seed)

    test_paths_df = dataset_paths.sample(frac=test_size, random_state=seed)
    train_paths_df = dataset_paths.drop(test_paths_df.index)

    train_dataset = DICDataset(paths_df=train_paths_df)
    test_dataset = DICDataset(paths_df=test_paths_df)

    print(f"[INFO] Training on {int(data_frac * 100)}% of samples.")
    print(f"[INFO] Total train samples: {len(train_dataset)}.")
    print(f"[INFO] Total test samples: {len(test_dataset)}.")

    train_dl = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
    )

    test_dl = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size,
    )

    print(f"[INFO] Train dataloader size: {len(train_dl)}")
    print(f"[INFO] Test dataloader size: {len(test_dl)}")

    return train_dl, test_dl

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
import yaml
import torch.nn as nn
from eval.eval_method import get_keyshot_summ


class VideoSumDataset(Dataset):
    def __init__(self, root, mode, split_id, batch_size, dataset_name):
        """
        Initialize dataset. A H5 file has the following attributes.

        {"change_points", "features", "gtscore", "gtsummary", "n_frame_per_seg",
        "n_steps", "picks", "user_summary", "video_name", "n_frames"}

        Args:
            root (str): dataset root.
            mode (str): "train" or "test".
            split_id (int): 0 - 4 (total 5 splits).
            batch_size (int): batch size.
            dataset_name (str): "SumMe" or "TVSum".
        """
        self.root = root
        self.batch_size = batch_size

        video_feature_path = Path(self.root, 'feature', f'eccv16_dataset_{dataset_name.lower()}_google_pool5.h5')
        text_feature_path = Path(self.root, 'feature', 'text_roberta.npy')
        split_yml_path = Path(self.root, 'splits.yml')

        with open(split_yml_path, 'r') as f:
            split_yml = yaml.safe_load(f)

        video_fea = h5py.File(video_feature_path)

        self.video_feature_dict = {}
        self.text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()

        self.org_keys = split_yml[split_id][f'{mode}_keys']

        for video_key in self.org_keys:
            key = video_key.split('/')[-1]
            fea = video_fea[key]
            fea_dict = {}
            for fea_key in fea.keys():
                fea_dict[fea_key] = np.array(fea[fea_key])
            self.video_feature_dict[key] = fea_dict

    def __len__(self):
        return len(self.org_keys)

    def __getitem__(self, index):

        video_id = self.org_keys[index].split('/')[-1]
        # get video_fea, text_fea, gt_summary and gt_score
        video = self.video_feature_dict[video_id]

        video_fea = torch.tensor(video['features'])
        text_fea = self.text_feature_dict[video_id]
        gt_summary = torch.tensor(video['gtsummary'])
        gt_score = torch.tensor(video['gtscore'])

        change_points = torch.tensor(video['change_points'])
        picks = torch.tensor(video['picks'])
        user_summary = torch.tensor(video['user_summary'])
        n_frame_per_seg = torch.tensor(video['n_frame_per_seg'])
        n_frames = torch.tensor(video['n_frames'])

        keyshot_summ, gtscore_upsampled = get_keyshot_summ(gt_score.numpy(), change_points.numpy(), n_frames.numpy(), n_frame_per_seg.numpy(), picks.numpy())
        target = torch.tensor(keyshot_summ[::15])

        # repeat text_fea util it has the same length with video_fea
        n_text = text_fea.shape[0]
        repeats = [16] * n_text
        repeats[n_text - 1] = (n_frames % 240) // 15 + 1
        text_fea = np.repeat(text_fea, repeats, axis=0)
        text_fea = torch.tensor(text_fea)

        # get mask
        mask = torch.ones(video_fea.shape[0])

        # padding the sequence

        return video_fea, text_fea, gt_score, gt_summary, mask, change_points, n_frames, n_frame_per_seg, picks, user_summary, target


def collate_fn(batch):
    video_fea = [item[0] for item in batch]
    text_fea = [item[1] for item in batch]
    gt_score = [item[2] for item in batch]
    gt_summary = [item[3] for item in batch]
    mask = [item[4] for item in batch]
    change_points = [item[5] for item in batch]
    n_frames = [item[6] for item in batch]
    n_frame_per_seg = [item[7] for item in batch]
    picks = [item[8] for item in batch]
    user_summary = [item[9] for item in batch]
    target = [item[10] for item in batch]

    # Padding 操作
    video_fea = pad_sequence(video_fea, batch_first=True)
    text_fea = pad_sequence(text_fea, batch_first=True)
    gt_score = pad_sequence(gt_score, batch_first=True)
    gt_summary = pad_sequence(gt_summary, batch_first=True)
    mask = pad_sequence(mask, batch_first=True)
    target = pad_sequence(target, batch_first=True)
    # change_points = pad_sequence(change_points, batch_first=True)
    # n_frames = pad_sequence(n_frames, batch_first=True)
    # n_frame_per_seg = pad_sequence(n_frame_per_seg, batch_first=True)
    # picks = pad_sequence(picks, batch_first=True)
    # user_summary = pad_sequence(user_summary, batch_first=True)

    return video_fea, text_fea, gt_score, gt_summary, mask, change_points, n_frames, n_frame_per_seg, picks, user_summary, target


class VSDataModule(LightningDataModule):
    def __init__(self, root="./data_source/SumMe", split_id=0, batch_size=4, dataset_name="SumMe"):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size

        self.train_set = VideoSumDataset(
            root=root,
            mode="train",
            split_id=split_id,
            batch_size=batch_size,
            dataset_name=dataset_name
        )

        self.val_set = VideoSumDataset(
            root=root,
            mode="test",
            split_id=split_id,
            batch_size=batch_size,
            dataset_name=dataset_name
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collate_fn)


if __name__ == '__main__':

    datamodule = VSDataModule(
        root='../data_source/SumMe',
        split_id=1,
        batch_size=4,
        dataset_name='SumMe'
    )
    data_loader = datamodule.train_dataloader()

    for batch in data_loader:
        video_fea, text_fea, gt_score, gt_summary, mask = batch
        mask = mask.bool()
        gt_score_unmasked = torch.masked_select(gt_score, mask)
        fake_score = torch.zeros_like(gt_score_unmasked)
        fake_score[0] = 1

        loss = nn.CrossEntropyLoss()

        loss_itm = loss(gt_score_unmasked, fake_score)
        print(batch.shape)

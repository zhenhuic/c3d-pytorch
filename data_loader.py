import cv2
import numpy as np
import torch
import linecache
import random
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, root, list_file, transform=None):
        """

        :param root:
        :param list_file:
        :param transform:
        """
        self.root = root
        self.list_file = list_file
        self.transform = transform

    def __getitem__(self, item):
        text = linecache.getline(self.list_file, item + 1)
        dt = text.split()
        video_dir, label = dt[0], dt[1]
        one_hot = np.zeros(101, dtype='float32')
        one_hot[int(label) - 1] = 1.0
        label = torch.from_numpy(one_hot)

        cap = cv2.VideoCapture(self.root + video_dir)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video = np.empty((frame_count, 112, 112, 3), np.dtype('float32'))
        fc, ret = 0, True
        while fc < frame_count and ret:
            ret, frame = cap.read()
            try:
                video[fc] = cv2.resize(frame, (112, 112))
            except Exception:
                continue
            fc += 1
        cap.release()

        start = int((frame_count - 16) * random.random())
        clip = video[start:(start + 16)]

        assert clip.shape == (16, 112, 112, 3)
        clip = torch.from_numpy(clip)
        # sample = {'clip': clip, 'label': label}
        sample = (clip, label)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        with open(self.list_file, 'r') as f:
            length = len(f.readlines())
            return length


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (clip, label).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (clip, label).
            - clip: torch tensor of shape (16, 112, 112, 3).
            - label: torch tensor of shape (101,).
    Returns:
        clips: torch tensor of shape (batch_size, 16, 112, 112, 3).
        targets: torch tensor of shape (batch_size, 101).
    """
    clips, labels = zip(*data)

    length = len(labels)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    clips = torch.stack(clips, 0)
    labels = torch.stack(labels, 0)

    return clips, labels


def get_loader(root, list_file, transform, batch_size, shuffle, num_workers):
    video_dataset = VideoDataset(root, list_file, transform)
    data_loader = DataLoader(dataset=video_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    dataset = VideoDataset(root='D:/Workspace/dataset/UCF-101/', list_file='ucfTrainTestlist/trainlist02.txt')

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['clip'].shape, sample['label'].shape)

        if i == 7:
            break

import abc
import pickle
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class Dataset(abc.ABC):
    def __init__(self, *args):
        self.data: np.ndarray = ...
        self.labels: np.ndarray = ...

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError


class DatasetCIFAR10(Dataset):
    LABELS_MAP = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    INVERSE_LABELS_MAP = {v: k for k, v in LABELS_MAP.items()}
    NUM_LABELS = len(LABELS_MAP)
    IMAGE_SIZE = (3, 32, 32)
    NUM_FILES = 5
    NUM_SAMPLES_PER_FILE = 10000

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as f:
            raw = pickle.load(f, encoding='bytes')
        return raw

    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """0 ~ 255 범위의 픽셀값을 갖는 이미지 데이터의 전형적인 normalization 방법"""
        data = (data / 127.5) - 1.0
        return data

    def load(self) -> None:
        num_channels, h, w = self.IMAGE_SIZE

        data, labels = [], []
        for i in range(1, self.NUM_FILES + 1):
            raw = self.unpickle(f"{self.data_path}/data_batch_{i}")
            _data: np.ndarray = raw[b'data']
            _data = _data.reshape(self.NUM_SAMPLES_PER_FILE, num_channels, h, w).astype(np.float64)
            _labels = np.array(raw[b'labels'])
            data.append(_data)
            labels.append(_labels)

        data = np.vstack(data)
        labels = np.concatenate(labels)

        self.data = self.normalize_data(data)
        self.labels = labels

    def show_index(self, index: int) -> None:
        img = self.data[index]
        label = self.labels[index]
        plot_img(img)
        print(f"Label: {self.INVERSE_LABELS_MAP.get(label)}")


def cvt_img(img: np.ndarray) -> np.ndarray:
    img = img - img.min()
    img = (img / img.max())
    return img.astype(np.float32)


def plot_img(img: Union[np.ndarray, torch.Tensor]) -> None:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_subplot()
    img = np.transpose(img, (1, 2, 0))  # RGB channel to the last dim
    ax.imshow(cvt_img(img))
    plt.show()
    plt.close()


def select_rgb_channel(img: np.ndarray, channel: str = None) -> np.ndarray:
    if channel is None:
        img_channel = img
    elif channel == "r":
        img_channel = img[:, 0]
    elif channel == "g":
        img_channel = img[:, 1]
    elif channel == "b":
        img_channel = img[:, 2]
    return img_channel

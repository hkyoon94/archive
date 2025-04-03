import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor as _T

SEED = 1
N_DATA = 5000
IN = 100
OUT = 5


def build_test_data() -> tuple[_T, _T]:
    x = torch.randn(N_DATA, 100).float()
    y = torch.zeros(N_DATA).long()
    for i in range(len(y)):
        y[i] = torch.randint(low=0, high=OUT, size=(1,))
    return x, y


class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Linear(IN, 150),
            nn.Linear(150, 200),
            nn.Linear(200, 150),
            nn.Linear(150, OUT),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: _T) -> _T:
        for layer in self.layers:
            x = x.to(layer.weight.device)
            x = layer(x)
        return x


def load_init(parent_pth: str) -> tuple[_T, _T, TestNN]:
    x_path = Path(parent_pth) / "x.pt"
    y_path = Path(parent_pth) / "y.pt"
    model_path = Path(parent_pth) / "model.pt"
    x = torch.load(str(x_path))
    y = torch.load(str(y_path))
    model_state = torch.load(str(model_path))
    model = TestNN()
    model.load_state_dict(model_state)
    model = model.train()
    return x, y, model


def get_batch_split(x: _T, y: _T, n_chunks: int, i: int) -> tuple[_T, _T]:
    x_chunk = x.chunk(n_chunks, dim=0)[i]
    y_chunk = y.chunk(n_chunks, dim=0)[i]
    return x_chunk, y_chunk


def barrier_delayed(global_rank: int) -> None:
    dist.barrier()
    time.sleep(global_rank * 0.2)
    print(f"global_rank {global_rank} ------------------------------ ")


def get_process_group_info() -> tuple[int, int, int, int]:
    world_size = int(os.environ["WORLD_SIZE"])  # multi-node에 걸쳐 열린 프로세스 수
    global_rank = int(os.environ["RANK"])  # global world_size 기준 현재 프로세스 번호
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])  # 특정 노드에 열린 프로세스 수
    local_rank = int(os.environ["LOCAL_RANK"])  # local world_size 기준 현재 프로세스 번호

    barrier_delayed(global_rank)
    print(f"world size: {world_size}")
    print(f"local_world_size: {local_world_size}")
    print(f"local_rank: {local_rank}")
    dist.barrier()
    return world_size, global_rank, local_world_size, local_rank


if __name__ == "__main__":
    # 데이터셋 & 모델 초기화 후 저장
    PARENT_DIR = "./prac/parallelism"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    x, y = build_test_data()
    model = TestNN()

    torch.save(x, f"{PARENT_DIR}/x.pt")
    torch.save(y, f"{PARENT_DIR}/y.pt")
    torch.save(model.state_dict(), f"{PARENT_DIR}/model.pt")

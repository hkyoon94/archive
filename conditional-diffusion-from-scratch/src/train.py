import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from yacs.config import CfgNode

from src.data_utils import Dataset, DatasetCIFAR10
from src.unet import UNetConditional


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./src/training_config.yml")
    parser.add_argument("--train_set_path", type=str, default="./cifar-10-batches-py")
    parser.add_argument("--work_dir", type=str, default="./checkpoints")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    return parser


class NoiseScaler:
    """이 부분이 핵심"""
    def __init__(self, num_steps: int, betas: np.ndarray):
        self.num_steps = num_steps

        # variance scheduling params
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = np.zeros_like(betas)  # i.e., noise ratio
        for i in range(self.betas.shape[0]):
            # alpha_bar_t = alpha_1 x alpha_2 x ... * alpha_t
            self.alpha_bars[i] = np.prod(self.alphas[:i + 1])

    def get_x_t_and_noise(self, x_0: np.ndarray, t: int) -> np.ndarray:
        noise = np.random.randn(*x_0.shape)
        # 주의: 구현상 alpha_bar의 인덱싱은 t 대신 t - 1 이어야 함
        x_t = np.sqrt(self.alpha_bars[t - 1]) * x_0 + np.sqrt(1 - self.alpha_bars[t - 1]) * noise
        return x_t, noise


class TrainLoader:
    def __init__(
        self,
        dataset: Dataset,
        noise_scaler: NoiseScaler,
        batch_size: int,
        random_step_per_batch: bool = False,
        seed: int = 7,
        shuffle: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.data = dataset.data.copy()
        self.labels = dataset.labels.copy()
        self.noise_scaler = noise_scaler
        self.num_steps = noise_scaler.num_steps
        self.bsz = batch_size
        self.random_step_per_patch = random_step_per_batch

        self.seed = seed
        self.shuffler = np.random.default_rng(seed)
        self.idx = np.array(range(self.data.shape[0]))
        self.num_data = len(self.labels)
        self.shuffle = shuffle
        self.device = device
        self.num_batches = None

    def __len__(self) -> int:
        if self.num_batches is None:
            self.num_batches = sum(1 for _ in self)
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            self.shuffler.shuffle(self.idx)
            self.data = self.data[self.idx]
            self.labels = self.labels[self.idx]

        for b_begin in range(0, self.num_data, self.bsz):
            b_end = min(b_begin + self.bsz, self.num_data)
            x_0s = self.data[b_begin: b_end]
            labels = self.labels[b_begin: b_end]
            t = np.random.randint(low=1, high=self.num_steps + 1)
            x_ts, noises = self.noise_scaler.get_x_t_and_noise(x_0=x_0s, t=t)
            steps = t * torch.ones(b_end - b_begin)

            yield (
                torch.from_numpy(x_ts).float().to(self.device),
                torch.from_numpy(noises).float().to(self.device),
                steps.float().to(self.device),
                torch.from_numpy(labels).long().to(self.device),
            )


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    # n_procs, gpu_id = 1, 3
    n_procs = int(os.environ['LOCAL_WORLD_SIZE'])
    assert len(args.gpus) == n_procs, "# gpus to use != --nproc_per_node"
    local_rank = int(os.environ['LOCAL_RANK'])
    gpu_id = args.gpus[local_rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    # WORKING DIRECTORY
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    # >>> PARAMETERS
    with open(args.config_path, "r") as f:
        cfg = CfgNode.load_cfg(f)
        
        batch_size = int(cfg.TRAIN.BATCH_SIZE / n_procs)
        beta_start = cfg.TRAIN.BETA_1
        beta_end = cfg.TRAIN.BETA_T
        num_steps = cfg.TRAIN.TOTAL_STEPS
        max_epoch = cfg.TRAIN.MAX_EPOCH
        ckpt_save_interval = cfg.TRAIN.SAVE_INTERVAL
        initial_lr = cfg.TRAIN.INITIAL_LR

        betas = np.linspace(beta_start, beta_end, num_steps)  # t=0 ~ t=num_steps
        
        def lr_schedule(step):
            return 0.99996 ** step + 0.0001
    # <<< PARAMETERS

    num_channels, height, width = DatasetCIFAR10.IMAGE_SIZE
    num_classes = DatasetCIFAR10.NUM_LABELS
    seed = 7 + gpu_id
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    # LOADING DATA
    train_set = DatasetCIFAR10(args.train_set_path)
    train_set.load()

    # INITIALIZING DATALOADER
    noise_scaler = NoiseScaler(betas=betas, num_steps=num_steps)
    train_loader = TrainLoader(
        dataset=train_set,
        noise_scaler=noise_scaler,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )
    print(f"Number of batches: {len(train_loader)}")

    # INITIALIZING MODEL
    num_channels = DatasetCIFAR10.IMAGE_SIZE[0]
    model = UNetConditional(
        cfg=cfg,
        c_in=num_channels,
        c_out=num_channels,
        num_classes=num_classes,
    )
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[gpu_id])
    dist.barrier()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
    scheuler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    criterion = torch.nn.MSELoss()

    # TRAINING
    epoch, step, lrs = 0, 0, []
    while True:
        epoch += 1
        epoch_losses = torch.zeros(len(train_loader)).float().to(device)
        train_loader = tqdm(train_loader, desc=f"Epoch: {epoch} ")
        for i, batch in enumerate(train_loader):
            step += 1
            xs, noise, steps, labels = batch
            out = model.forward(x=xs, t=steps, y=labels)
            loss = criterion.forward(out, noise)
            loss.backward()
            optimizer.step()
            scheuler.step()
            optimizer.zero_grad()

            # logging
            loss_value = loss.item()
            epoch_losses[i] = loss_value
            dist.all_reduce(epoch_losses, op=dist.ReduceOp.SUM)
            epoch_losses /= n_procs
            train_loader.set_postfix_str(
                f"Step: {step} | loss: {loss_value:.6f} | lr: {scheuler.get_last_lr()[0]:.6f}"
            )
            dist.barrier()

        if local_rank == 0:
            print(f"Epoch mean loss: {epoch_losses.mean().item():.8f}")
            if epoch % ckpt_save_interval == 0:  # SAVING CKPT
                torch.save(model.module.state_dict(), f"{work_dir}/epoch_{epoch}.pt")
        if epoch == max_epoch:
            break
    
    dist.destroy_process_group()

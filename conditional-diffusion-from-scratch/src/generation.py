import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.data_utils import select_rgb_channel
from src.unet import UNet, UNetConditional
from src.train import NoiseScaler


class Preprocessor:
    def __init__(self, labels_map: dict[str, int]):
        self.labels_map = labels_map

    def preprocess(self, label: str = "automobile") -> int:
        """레이블 인코딩"""
        enc_label = self.labels_map.get(label)
        print(f"Encoded label #: {enc_label}")
        if enc_label is None:
            raise ValueError("Unknown conditional label!")
        return enc_label


class Predictor:
    def __init__(
        self,
        model: Union[UNet, UNetConditional],
        betas: np.ndarray,
        imsize: tuple[int],
        device: torch.device,
    ):
        self.model = model
        self.num_steps = len(betas)
        noise_scaler = NoiseScaler(num_steps=self.num_steps, betas=betas)
        self.betas = noise_scaler.betas
        self.alphas = noise_scaler.alphas
        self.alpha_bars = noise_scaler.alpha_bars

        self.imsize = imsize
        self.device = device

    # DDPM ========================================================================================
    def ddpm_reverse_step(
        self,
        x: np.ndarray,
        t: int,
        a: float,  # alpha_t
        a_bar: float,  # alpha_bar_t
        b: float,  # beta_t
        y: Tensor,  # conditional embedding
        bsz: int,  # batch size
    ) -> np.ndarray:
        out = self.model.forward(  # 논문의 \epsilon_\theta(x_t, t, y)와 같음
            x=torch.from_numpy(x).float().to(self.device),
            t=t * torch.ones(bsz).float().to(self.device),
            y=y,
        ).cpu().numpy()

        z = np.random.randn(*out.shape) if t != 1 else 0
        x = 1 / np.sqrt(a) * (x - ((1 - a) / (np.sqrt(1 - a_bar)) * out)) + np.sqrt(b) * z
        # 주의: 구현상 beta, alpha, alpha_bar의 인덱싱은 t 대신 t - 1 이어야 함
        return x

    @torch.inference_mode(True)
    def ddpm_predict(self, ts: np.ndarray, enc_label: int) -> list[Tensor]:
        """단일 initial noise로부터 step별 denoising 과정을 모두 x_hist 변수에 저장"""
        x_hist = []
        x = np.expand_dims(np.random.randn(*self.imsize), axis=0)  # x_T: 단일 batch이므로 unsqueeze
        bsz = 1
        y = enc_label * torch.ones(1).long().to(self.device)
        x_hist.append(x[0])

        ts = ts[::-1]  # reverse t
        for t in tqdm(ts[:-1]):  # reverse process (from ts[-1] to 1)
            a = self.alphas[t - 1]
            a_bar = self.alpha_bars[t - 1]
            b = self.betas[t - 1]
            x = self.ddpm_reverse_step(x, t, a, a_bar, b, y=y, bsz=bsz)
            x_hist.append(x[0])

        assert len(x_hist) == self.num_steps + 1
        return x_hist  # x_T:0

    @torch.inference_mode(True)
    def ddpm_predict_parallel(self, ts: np.ndarray, enc_label: int, n: int = 25) -> Tensor:
        """n개의 initial noise로부터 병렬 생성"""
        x = np.random.randn(n, *self.imsize)  # x_T
        bsz = x.shape[0]
        y = enc_label * torch.ones(bsz).long().to(self.device)

        ts = ts[::-1]
        for t in tqdm(ts[:-1]):
            a = self.alphas[t - 1]
            a_bar = self.alpha_bars[t - 1]
            b = self.betas[t - 1]
            x = self.ddpm_reverse_step(x, t, a, a_bar, b, y=y, bsz=bsz)

        return x  # x_0

    # DDIM ========================================================================================
    def ddim_reverse_step(
        self,
        x: np.ndarray,
        t: int,
        a_t: float,  # alpha_bar_t
        a_tm1: int,  # alpha_bar_{t-1}
        eta: float,
        y: Tensor,
        bsz: int,
    ) -> np.ndarray:
        out = self.model.forward(
            x=torch.from_numpy(x).float().to(self.device),
            t=t * torch.ones(bsz).float().to(self.device),
            y=y,
        ).cpu().numpy()

        sigma = eta * np.sqrt((1 - a_tm1) / (1 - a_t)) * np.sqrt(1 - (a_t / a_tm1))
        z = np.random.randn(*out.shape) if eta is not 0 else 0  # if eta=0, then perfect DDIM
        x = np.sqrt(a_tm1) * (
            (x - np.sqrt(1 - a_t) * out) / np.sqrt(a_t)
        ) + np.sqrt(1 - a_tm1 - sigma ** 2) * out + sigma * z

        return x

    @torch.inference_mode(True)
    def ddim_predict(self, ts: np.ndarray, eta: float, enc_label: int) -> list[Tensor]:
        x_hist = []
        x = np.expand_dims(np.random.randn(*self.imsize), axis=0)
        bsz = 1
        y = enc_label * torch.ones(1).long().to(self.device)
        x_hist.append(x[0])
        
        ts = ts[::-1]  # reverse t
        # >>> 중요: DDIM은 t=0일 때의 alpha_bar 값이 필요하므로, 이론상 alpha_bar_0인 1을 넣어줌.
        alpha_bars = np.insert(self.alpha_bars, 0, 1)

        for t, tm1 in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1):
            a_t = alpha_bars[t]
            a_tm1 = alpha_bars[tm1]
            x = self.ddim_reverse_step(x, t, a_t, a_tm1, eta=eta, y=y, bsz=bsz)
            x_hist.append(x[0])

        assert len(x_hist) == len(ts)
        return x_hist  # x_T:0

    @torch.inference_mode(True)
    def ddim_predict_parallel(self, ts: np.ndarray, eta: float, enc_label: int, n: int = 25) -> Tensor:
        """n개의 initial noise로부터 병렬 생성"""
        x = np.random.randn(n, *self.imsize)  # x_T
        bsz = x.shape[0]
        y = enc_label * torch.ones(bsz).long().to(self.device)

        ts = ts[::-1]  # reverse t
        alpha_bars = np.insert(self.alpha_bars, 0, 1)

        for t, tm1 in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1):
            a_t = alpha_bars[t]
            a_tm1 = alpha_bars[tm1]
            x = self.ddim_reverse_step(x, t, a_t, a_tm1, eta=eta, y=y, bsz=bsz)

        return x  # x_0


class Postprocessor:
    def __init__(self, num_steps: Optional[int] = None):
        self.num_steps = num_steps

    @staticmethod
    def cvt_img(img: np.ndarray) -> np.ndarray:
        img = img - img.min()
        img = (img / img.max())
        return img.astype(np.float32)

    def plot_image(self, ax: plt.Axes, img: Union[np.ndarray, torch.Tensor]) -> plt.Axes:
        if isinstance(img, torch.Tensor):
            img = img.squeeze(dim=0).detach().cpu().numpy()
        if len(img.shape) == 3:  # if has RGB channel
            img = np.transpose(img, (1, 2, 0))  # RGB channel to the last dim
        ax.imshow(self.cvt_img(img))
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        return ax

    def postprocess(
        self,
        x_hist: list[torch.Tensor],
        ts: np.ndarray,
        stride: int,
        fig_size: tuple[int, int],
        channel: str = "",
    ) -> None:
        """step별 생성물을 stride마다 플롯"""
        fig = plt.figure(figsize=fig_size)
        ncols = 4
        ts = ts[::stride]
        if 0 not in ts:
            ts = np.append(ts, 0)
        x_hist = np.array(x_hist, dtype=object)[::stride]

        len_ts = n_plots = len(ts)
        for i, x in enumerate(x_hist):
            ax = fig.add_subplot(math.ceil((n_plots + 1) / ncols), ncols, i + 1)
            img_channel = select_rgb_channel(img=x, channel=channel)
            ax = self.plot_image(ax, img_channel)
            ax.set_title(f"step: {ts[len_ts - 1 - i]}", fontsize=10)
        fig.tight_layout()
        plt.show()
        plt.close()

    def postprocess_parallel(
        self,
        x: torch.Tensor,
        fig_size: tuple[int, int],
        channel: str = "",
    ) -> None:
        """ 병렬 batch 생성을 한꺼번에 플롯 """
        fig = plt.figure(figsize=fig_size)
        ncols = 5
        for b, img in enumerate(x):
            ax = fig.add_subplot(math.ceil((x.shape[0]) / ncols), ncols, b + 1)
            img_channel = select_rgb_channel(img, channel=channel)
            ax = self.plot_image(ax, img_channel)
            ax.set_title(f"batch {b}", fontsize=10)
        fig.tight_layout()
        plt.show()
        plt.close()


class Generator:
    def __init__(
        self,
        preprocessor: Preprocessor,
        predictor: Predictor,
        postprocessor: Postprocessor = Postprocessor(),
    ):
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor

        self.postprocessor.num_steps = self.predictor.num_steps

    def generate(
        self,
        label: str = "automobile",
        method: str = "ddpm",  # "'ddpm' or 'ddim'"
        ts: Optional[np.ndarray] = None,  # timesteps
        eta: Optional[float] = None,  # 
        stride: int = 1,  # stride number for plotting
        fig_size: tuple[int, int] = (8, 10),
        channel: Optional[str] = None,
    ) -> list[torch.Tensor]:
        """
        한 이미지 생성 과정을 스텝별로 관찰.
        params:
            label: str: CIFAR-10'의 이미지 레이블 (ex: "airplane").
            method: str: Denoising 기법 ('ddpm' 또는 'ddim').
            ts: Optional[np.ndarray]: Denoising 과정에 사용할 타임스텝.
                <method> 변수가 'ddpm'일 경우, 무시됨.
            eta: float: 랜덤 노이즈 사용 정도를 조절하는 파라미터. (0일 경우 완벽한 DDIM)
            stride: int: 플롯 시 time-step을 얼마나 건너뛸 지 정하는 파라미터.
            fig_size: tuple[int, int]: 최종 이미지 플롯 결과물 크기 (가로 x 세로 (inches))
            channel: Optional[str]: 최종 이미지 플롯에 표시할 RGB 채널 ('r' or 'g' or 'b')
                None일 경우 RGB 채널 전체를 플롯.
        return:
            x_hist: ts별 Denoised 이미지들의 히스토리.
        """
        print(f"Generating label {label}...")
        enc_label = self.preprocessor.preprocess(label=label)
        if ts is None:
            ts = np.arange(self.predictor.num_steps + 1)
        if method == "ddpm":
            x_hist = self.predictor.ddpm_predict(
                ts=ts,
                enc_label=enc_label,
            )
        elif method == "ddim":
            if eta is None:
                raise ValueError("Specify 'eta' to use DDIM sampling")
            x_hist = self.predictor.ddim_predict(
                ts=ts,
                eta=eta,
                enc_label=enc_label
            )
        else:
            raise ValueError(f"Unknown sampling method: '{method}'")
        self.postprocessor.postprocess(
            x_hist,
            ts=ts,
            stride=stride,
            fig_size=fig_size,
            channel=channel,
        )
        return x_hist

    def generate_parallel(
        self,
        label: str = "automobile",
        method: str = "ddpm",
        n: int = 25,
        ts: Optional[np.ndarray] = None,
        eta: float = 1,
        fig_size: tuple[int, int] = (8, 9),
        channel: str = None,
    ) -> None:
        """
        이미지들을 n개 만큼 한꺼번에 병렬 생성.
        params:
            label: str: CIFAR-10'의 이미지 레이블 (ex: "airplane").
            method: str: Denoising 기법 ('ddpm' 또는 'ddim').
            n: 병렬 생성할 이미지 갯수.
            ts: Optional[np.ndarray]: Denoising 과정에 사용할 타임스텝.
                <method> 변수가 'ddpm'일 경우, 무시됨.
            eta: float: 랜덤 노이즈 사용 정도를 조절하는 파라미터. (0일 경우 완벽한 DDIM)
            fig_size: tuple[int, int]: 최종 이미지 플롯 결과물 크기 (가로 x 세로 (inches))
            channel: Optional[str]: 최종 이미지 플롯에 표시할 RGB 채널 ('r' or 'g' or 'b')
                None일 경우 RGB 채널 전체를 플롯.
        return:
            xs: batch 별 최종 Denoised 이미지 (t=0).
        """
        print(f"Generating {n} images of label {label}...")
        enc_label = self.preprocessor.preprocess(label=label)
        
        if method == "ddpm":
            ts = np.arange(self.predictor.num_steps + 1)
            xs = self.predictor.ddpm_predict_parallel(
                ts=ts,
                n=n,
                enc_label=enc_label,
            )
        elif method == "ddim":
            if eta is None:
                raise ValueError("Specify 'eta' to use DDIM sampling")
            xs = self.predictor.ddim_predict_parallel(
                ts=ts,
                eta=eta,
                enc_label=enc_label,
                n=n,
            )
        else:
            raise ValueError(f"Unknown sampling method: '{method}'")
        self.postprocessor.postprocess_parallel(
            xs,
            fig_size=fig_size,
            channel=channel,
        )
        return xs

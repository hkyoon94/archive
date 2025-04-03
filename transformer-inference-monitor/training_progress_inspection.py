import argparse
import multiprocessing as mp
import os
import queue
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from midiai.distributed import DistributedProcessPoolExecutor
from midiai.inspector.monitoring.interpretation import InterpretedSequence, SequenceInterpreter
from midiai.inspector.monitoring.vocab_utils import TokenFields as TF
from midiai.inst_gen.vocab import InstGenVocab
from midiai.inst_gen.vocab.maps import MetaEncodingMap
from midiai.logger import logger

# numpy의 멀티쓰레딩 해제: 멀티 프로세싱시 개별 프로세스 내에서 오히려 병목 발생
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

INVERSE_GENRE_MAP = {v: k for k, v in MetaEncodingMap.GENRE_MAP.items()}
INVERSE_TRACK_CATEGORY_MAP = {v: k for k, v in MetaEncodingMap.TRACK_CATEGORY_MAP.items()}

GENRE_OFFSET = InstGenVocab.meta_word.GENRE.value.offset
TRACK_CATEGORY_OFFSET = InstGenVocab.meta_word.TRACK_CATEGORY.value.offset


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, default=None, help="run을 비교할 baseline 디렉토리.")
    parser.add_argument(
        "--extract_baseline_stats",
        action="store_true",
        help="False 일시 baseline에 대한 통계 추출을 진행하지 않음",
    )
    parser.add_argument(
        "--baseline_processed_data_dir",
        type=str,
        default=None,
        help=(
            "--extract_baseline_stats가 True일 경우, 통계 추출을 실시할 전처리된 데이터 디렉터리: "
            "None 일시 '<--exp_dir>/data'로 고정."
        ),
    )
    parser.add_argument("--exp_dir", type=str, default=None, help="baseline과 비교할 실험을 진행한 디렉터리.")
    parser.add_argument(
        "--extract_exp_stats",
        action="store_true",
        help="False 일시 experiment에 대한 통계 추출을 진행하지 않음",
    )
    parser.add_argument(
        "--exp_processed_data_dir",
        type=str,
        default=None,
        help=(
            "--extract_exp_stats가 True일 경우, 통계 추출을 실시할 전처리된 데이터 디렉터리: "
            "None 일시 '<--exp_dir>/data'로 고정."
        ),
    )
    parser.add_argument(
        "--comparison_plot",
        action="store_true",
        help="comparison plot을 진행할 지 말지 결정하는 인자.",
    )
    parser.add_argument("--raw_data_dir", type=str, default="/server2/disk2/data/raw_data")
    parser.add_argument("--use_augmented", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--n_workers", type=int, default=None)
    return parser


class CheckpointPath(Path):
    """체크포인트 저장 경로명을 파싱하기 위한 Path type 상속 객체"""

    _flavour = type(Path())._flavour  # path 상속에 필요한 구문

    @property
    def step(self) -> int:
        return int(self.name.split("_")[1].split("-")[0])

    @property
    def train_loss(self) -> float:
        return float(self.name.split("_")[2].split("-")[0])

    @property
    def val_loss(self) -> Optional[float]:
        try:
            return float(self.name.split("_")[3].rstrip(".pt"))
        except IndexError:
            return None


class StatisticsPath(Path):
    """통계치 csv의 경로를 파싱하기 위한 Path type 상속 객체"""

    _flavour = type(Path())._flavour  # path 상속에 필요한 구문

    @property
    def step(self) -> int:
        return int(self.name.split(".")[0])


class InstGenTrainingResultsPath:
    """
    Inst-gen의 training progress inspection을 위한 클래스.
    self.STATS_DIR_NAME 이름의 하위 디렉터리에 저장된 statistics를 불러와
    분석을 위한 시각화를 진행함.
    """

    STATS_DIR_NAME = "stats"  # 약속된 하위 통계 추출 디렉터리 명

    def __init__(self, path: str):
        self.path = Path(path)

        _ckpt_paths = glob(f"{path}/*.pt")
        self.ckpt_paths: list[CheckpointPath] = []
        for ckpt_path in _ckpt_paths:
            if "best" not in ckpt_path and "last" not in ckpt_path:
                self.ckpt_paths.append(CheckpointPath(ckpt_path))
        self.ckpt_paths.sort(key=lambda x: x.step)
        try:
            self.ckpt_best_path = CheckpointPath(glob(f"{path}/best*.pt")[0])
        except IndexError:
            self.ckpt_best_path = None
        try:
            self.ckpt_last_path = CheckpointPath(glob(f"{path}/last*.pt")[0])
        except IndexError:
            self.ckpt_best_path = None

        self.stats_dir = f"{path}/{self.STATS_DIR_NAME}"

        self.stat_paths: list[StatisticsPath] = []
        for stat_path in glob(f"{self.stats_dir}/*.csv"):
            self.stat_paths.append(StatisticsPath(stat_path))
        self.stat_paths.sort(key=lambda x: x.step)

    @property
    def ckpt_steps(self) -> list[int]:
        return [ckpt_path.step for ckpt_path in self.ckpt_paths]

    @property
    def ckpt_train_losses(self) -> np.ndarray:
        return np.array([ckpt_path.train_loss for ckpt_path in self.ckpt_paths])

    @property
    def ckpt_val_losses(self) -> np.ndarray:
        return np.array([ckpt_path.val_loss for ckpt_path in self.ckpt_paths])

    @property
    def min_step_val_loss(self) -> tuple[int, float]:
        """훈련과정 중 minimum val-loss가 찍힌 step과 해당 loss값을 리턴"""
        min_index = self.ckpt_val_losses.argmin()
        return self.ckpt_steps[min_index], self.ckpt_val_losses[min_index]

    @property
    def stat_steps(self) -> list[int]:
        return [stat_path.step for stat_path in self.stat_paths]

    @property
    def stat_dfs(self) -> list[pd.DataFrame]:
        return [pd.read_csv(str(stat_path), index_col=False) for stat_path in self.stat_paths]


def statistics_extraction_subprocess(
    proc_id: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    logging_path: str,
    *init_args,
) -> None:
    """
    체크포인트 번호를 받아 val-set statistics extraction을 수행하는 ParallelProcessPool용 태스크
    """

    def load_sequences(info_seq_paths: list[str], note_seq_paths: list[str]) -> np.ndarray:
        len_seqs = len(info_seq_paths)
        seqs = np.empty(len_seqs, dtype=object)
        ids, _seqs = [], []
        for info_seq_path, note_seq_path in zip(info_seq_paths, note_seq_paths):
            ids.append(info_seq_path.split("/")[-1].split("_")[-1].split(".")[0])
            info_seq = np.load(info_seq_path, allow_pickle=True).squeeze().tolist()
            note_seq = np.load(note_seq_path, allow_pickle=True).squeeze().tolist()
            _seqs.append([0] + info_seq + note_seq)
        seqs[:] = _seqs

        return ids, seqs

    def parse_genre_and_track_category(genre_id: int, track_category_id: int) -> tuple[str, str]:
        genre = INVERSE_GENRE_MAP[genre_id - GENRE_OFFSET]
        track_category = INVERSE_TRACK_CATEGORY_MAP[track_category_id - TRACK_CATEGORY_OFFSET]
        return genre, track_category

    def extract_iseq_stats(iseq: InterpretedSequence) -> dict[str, float]:
        # computing nlls
        """토큰 카테고리 별 기본 예측 성능 척도."""
        nll_notes = iseq.stepwise_nll(values_only=True)
        nll_bar = iseq.stepwise_nll(TF.BAR, values_only=True)
        nll_position = iseq.stepwise_nll(TF.NOTE_POSITION, values_only=True)
        nll_velocity = iseq.stepwise_nll(TF.VELOCITY, values_only=True)
        nll_pitch = iseq.stepwise_nll(TF.PITCH, values_only=True)
        nll_duration = iseq.stepwise_nll(TF.NOTE_DURATION, values_only=True)
        nll_chord = iseq.stepwise_nll(TF.CHORD, values_only=True)

        # computing entropies
        """생성시 diversity의 '매우' 간접적인 척도."""
        ent = iseq.stepwise_entropy(values_only=True)
        ent_notes = iseq.stepwise_entropy(target_field=TF.NOTES, values_only=True)

        # computing attention coherences
        """
        어텐션 맵 집중도 메트릭
        (예: conditional(메타, 가이드) 토큰들의 어텐션 반영도..? 등)
        """
        ...  # TODO: 아이디어 구상

        # computing harmonic coherences
        """
        (예: 한 코드 토큰이 지속될 경우 등장하는 pitch확률들의 코드 톤 응집도..? 등)
        """
        ...  # TODO: 아이디어 구상

        return {
            "nll_notes": nll_notes.mean().item(),
            "nll_bar": nll_bar.mean().item(),
            "nll_position": nll_position.mean().item(),
            "nll_velocity": nll_velocity.mean().item(),
            "nll_pitch": nll_pitch.mean().item(),
            "nll_chord": nll_chord.mean().item(),
            "nll_duration": nll_duration.mean().item(),
            "ent": ent.mean().item(),
            "ent_notes": ent_notes.mean().item(),
        }

    logger.set_name(f"worker {proc_id}")
    logger.set_logging_path(logging_path)
    cuda_id, _ = proc_id.split("-")
    processed_data_dir, augmented, stats_dir = init_args
    os.makedirs(stats_dir, exist_ok=True)

    logger.info("Loading sequences ...")
    if not augmented:
        info_seq_paths = glob(f"{processed_data_dir}/val/raw/info_*")
        note_seq_paths = glob(f"{processed_data_dir}/val/raw/note_*")
    else:  # TODO: raw/ 대신 augmented/ 이용 분기 작성
        ...
    info_seq_paths.sort()
    note_seq_paths.sort()
    ids, seqs = load_sequences(info_seq_paths, note_seq_paths)

    output_queue.put(proc_id)

    while True:
        try:
            task_idx, task_input = input_queue.get()
        except queue.Empty:
            continue

        ckpt_path: CheckpointPath = task_input
        stats_save_path = f"{stats_dir}/{ckpt_path.step}.csv"

        interpreter = SequenceInterpreter(
            checkpoint_path=ckpt_path,
            map_location=cuda_id,
            return_attns=False,
            lighter_mode=True,  # 토큰 히스토리 파싱을 건너뛰어 약간의 속도 향상
        )
        interpreter.set_sampling_func(sampling_func=None)

        stats_all = []
        for id, seq in tqdm(zip(ids, seqs), total=len(ids), mininterval=0.5):
            genre, track_category = parse_genre_and_track_category(
                genre_id=seq[8],  # TODO: index 하드코딩 제거..? (maps.py의 순서에서 읽어옴)
                track_category_id=seq[9],
            )
            stats = {
                "id": id,
                "genre": genre,
                "track_category": track_category,
            }
            iseq = interpreter.interpret(sequence=seq)
            stats |= extract_iseq_stats(iseq=iseq)
            stats_all.append(stats)

        pd.DataFrame(stats_all).to_csv(stats_save_path, sep=",", index=False)

        logger.info(f"{proc_id} finished task {task_idx}")
        output_queue.put((task_idx, None))


class InstGenTrainingStatsComparisonPlotter:
    BASE_PLOT_COLOR = "k"
    EXP_PLOT_COLOR = "r"
    SUBPLOT_SIZE = (3, 3)

    def __init__(
        self,
        baseline_dir: InstGenTrainingResultsPath,
        exp_dir: InstGenTrainingResultsPath,
    ):
        """
        self.features의 값들은, id, genre, track_category등의 메타범주를 제외한,
        extraction_subprocess로부터 추출된 통계 범주.
        ex: "nll_notes", "nll_pitch", ... etc.
        """
        self.baseline_dir = baseline_dir
        self.exp_dir = exp_dir

        self.baseline_dfs = self.baseline_dir.stat_dfs
        self.exp_dfs = self.exp_dir.stat_dfs

        self.features = self.baseline_dfs[0].columns.drop(["id", "genre", "track_category"])

    @staticmethod
    def get_field_values(field_name: str, field: str, dfs: list[pd.DataFrame]) -> np.ndarray:
        for i, df in enumerate(dfs):
            # TODO: 이렇게 stack해서 어레이로 리턴하지 말고, 매핑으로 리턴 ..?
            df_field = df.query(f"{field_name} == @field")
            values = df_field.mean(numeric_only=True).values
            if i == 0:
                stacked_values = values
            else:
                stacked_values = np.vstack((stacked_values, values))
        return stacked_values

    def plot_field_comparision(self, field_name: str) -> None:
        """baseline_dir와 exp_dir의 statistics를 field / feature 별로 비교 플롯."""
        logger.info(f"Drawing '{field_name}' comparison plots ...")
        fields = list(set(self.baseline_dfs[0][field_name].values))
        fields.sort()

        nrows = len(fields)
        ncols = len(self.features)
        subplot_horz_size, subplot_vert_size = self.SUBPLOT_SIZE
        fig = plt.figure(figsize=(subplot_horz_size * ncols, subplot_vert_size * nrows))
        gs = GridSpec(nrows, ncols)

        for row, field in tqdm(enumerate(fields), total=len(fields)):
            base_values = self.get_field_values(field_name, field, self.baseline_dfs)
            exp_values = self.get_field_values(field_name, field, self.exp_dfs)

            for col, feature in enumerate(self.features):
                ax = fig.add_subplot(gs[row, col])

                base_plot_values = base_values[:, col]
                ax.plot(  # baseline feature plot
                    self.baseline_dir.stat_steps,
                    base_plot_values,
                    c=self.BASE_PLOT_COLOR,
                )
                base_min_index = base_plot_values.argmin()
                base_min = base_plot_values[base_min_index]
                base_min_step = self.baseline_dir.stat_steps[base_min_index]
                ax.plot(  # plotting baseline minimum position bar
                    [base_min_step, base_min_step],
                    [base_min - 0.25, base_min + 0.25],
                    c=self.BASE_PLOT_COLOR,
                    linewidth=1.5,
                )

                exp_plot_values = exp_values[:, col]
                ax.plot(  # experiment feature plot
                    self.exp_dir.stat_steps,
                    exp_plot_values,
                    c=self.EXP_PLOT_COLOR,
                )
                exp_min_index = exp_plot_values.argmin()
                exp_min = exp_plot_values[exp_min_index]
                exp_min_step = self.exp_dir.stat_steps[exp_min_index]
                ax.plot(  # plotting experiment minimum position bar
                    [exp_min_step, exp_min_step],
                    [exp_min - 0.25, exp_min + 0.25],
                    c=self.EXP_PLOT_COLOR,
                    linewidth=1.5,
                )

                if feature.startswith("nll"):  # nll 관련 척도인 경우, overall val nll 정보를 플롯
                    ax.plot(  # baseline validation nll plot
                        self.baseline_dir.ckpt_steps,
                        self.baseline_dir.ckpt_val_losses,
                        "--",
                        linewidth=0.5,
                        c=self.BASE_PLOT_COLOR,
                    )
                    base_min_step, base_min_nll = self.baseline_dir.min_step_val_loss
                    ax.scatter(base_min_step, base_min_nll, s=10, c=self.BASE_PLOT_COLOR)

                    ax.plot(  # experiment validation nll plot
                        self.exp_dir.ckpt_steps,
                        self.exp_dir.ckpt_val_losses,
                        "--",
                        linewidth=0.5,
                        c=self.EXP_PLOT_COLOR,
                    )
                    exp_min_step, exp_min_nll = self.exp_dir.min_step_val_loss
                    ax.scatter(exp_min_step, exp_min_nll, s=10, c=self.EXP_PLOT_COLOR)

                ax.set_ylabel(field)
                ax.set_title(feature)
                ax.grid(True)

        logger.info("Saving figure ...")
        fig.tight_layout()
        fig.savefig(f"{self.exp_dir.path}/comparison_plot-{field_name}.png")
        plt.close()


def main(args: argparse.Namespace) -> None:
    # 프로세스는 gpu당 1개씩 (이 경우는 gpu당 프로세스 여러개보다 이게 가장 오버헤드 없이 빠름)
    n_workers = len(args.gpus) if args.n_workers is None else args.n_workers

    baseline_dir = InstGenTrainingResultsPath(args.baseline_dir)
    if args.baseline_processed_data_dir is None:
        baseline_processed_data_dir = baseline_dir.path.parent.parent / "data"
    else:
        baseline_processed_data_dir = args.baseline_processed_data_dir

    # extracting baseline statitistics
    if args.extract_baseline_stats:
        with DistributedProcessPoolExecutor(
            distributed_task=statistics_extraction_subprocess,
            task_init_args=(
                baseline_processed_data_dir,
                args.use_augmented,
                baseline_dir.stats_dir,
            ),
            num_procs=n_workers,
            gpu_ids=args.gpus,
            logging_path=baseline_dir.stats_dir,
        ) as executor:
            executor.push(baseline_dir.ckpt_paths)

    exp_dir = InstGenTrainingResultsPath(args.exp_dir)
    if args.exp_processed_data_dir is None:
        exp_processed_data_dir = exp_dir.path.parent.parent / "data"
    else:
        exp_processed_data_dir = args.exp_processed_data_dir

    # extracting experiment statitistics
    if args.extract_exp_stats:
        with DistributedProcessPoolExecutor(
            distributed_task=statistics_extraction_subprocess,
            task_init_args=(
                exp_processed_data_dir,
                args.use_augmented,
                exp_dir.stats_dir,
            ),
            num_procs=n_workers,
            gpu_ids=args.gpus,
            logging_path=exp_dir.stats_dir,
        ) as executor:
            executor.push(exp_dir.ckpt_paths)

    # collecting & comparison plotting extracted statistics
    if args.comparison_plot:
        comparison_plotter = InstGenTrainingStatsComparisonPlotter(
            baseline_dir=baseline_dir,
            exp_dir=exp_dir,
        )
        comparison_plotter.plot_field_comparision(field_name="genre")
        comparison_plotter.plot_field_comparision(field_name="track_category")


if __name__ == "__main__":
    known_args, _ = get_parser().parse_known_args()
    main(known_args)

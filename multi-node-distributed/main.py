import argparse
import time

import torch
import torch.nn as nn
from torch import Tensor as _T
from tqdm import tqdm

from base import (
    TestNN,
    barrier_delayed,
    get_batch_split,
    get_process_group_info,
    load_init,
)

torch.set_printoptions(precision=4, sci_mode=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="./",
    )
    parser.add_argument("--run_type", type=str, default="basic")
    parser.add_argument("--num_loops", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    return parser


def run_basic(
    x: _T,
    y: _T,
    model: TestNN,
    crit: nn.CrossEntropyLoss,
    num_loops: int,
) -> None:
    """
    기본적인 싱글프로세스 - 단일 gpu 학습
    """
    print("Running basic pipeline:")
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    loss_sum = 0
    st = time.time()
    for _ in tqdm(range(num_loops)):
        optimizer.zero_grad()
        x_out = model.forward(x.to(device))
        loss = crit.forward(x_out, y.to(device))
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
    ft = time.time()

    print(f"total loss: {loss_sum}")
    print(f"First layer weight grad:\n{model.layers[0].weight.grad}")
    print(f"Elapsed: {ft - st:8f} total | {(ft - st) / num_loops:8f}/epoch")


def run_ga(  # gradient accumulation
    x: _T,
    y: _T,
    model: TestNN,
    crit: nn.CrossEntropyLoss,
    num_loops: int,
) -> None:
    """
    Batch chunking 후 기울기 누적을 이용해 가장 VRAM소모를 적게 훈련시키는 방법
        -> 시간 소모량 상승
    """
    print("Running GA pipeline:")
    device = torch.device("cuda:3")
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    n_chunks = 11
    x_chunks, y_chunks = x.chunk(n_chunks, dim=0), y.chunk(n_chunks, dim=0)

    loss_sum = 0
    st = time.time()
    for i in tqdm(range(num_loops)):
        optimizer.zero_grad()
        for x, y in zip(x_chunks, y_chunks):
            x_out = model.forward(x.to(device))
            # grad 누적을 위해 loss값을 청크 수만큼 나눠줌
            loss = crit.forward(x_out, y.to(device)) / n_chunks
            loss.backward()  # optimizer.step() 전에 루프 안에서 반복적으로 backward() 호출
            loss_sum += loss.item()
        optimizer.step()
    ft = time.time()

    print(f"total loss: {loss_sum}")
    print(f"First layer weight grad:\n{model.layers[0].weight.grad}")
    print(f"Elapsed: {ft - st:8f} total | {(ft - st) / num_loops:8f}/epoch")


def run_mp(  # model parallel
    x: _T,
    y: _T,
    model: TestNN,
    crit: nn.CrossEntropyLoss,
    num_loops: int,
) -> None:
    """
    Model parallelism: 모델의 파라미터를 device별로 분산하여 컴퓨팅
        - 모델 초기화 후, 모델의 각 레이어들을 적절하게 서로 다른 device로 분산
        - .forward() 도중 배치텐서들의 .to()를 메서드를 활용하여
            다음에 들어갈 레이어들의 device로 보내주는 일부 구문 작성이 필수

    nn.Module 객체는 .to() 메서드를 적용시, 리턴값을 받지 않아도 내부 텐서들이 모두 해당 device에 등록된다.
    """
    print("Running MP pipeline:")
    for i, layer in enumerate(model.layers):  # layer별로 다른 device에 deploy
        layer.to(f"cuda:{i}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    loss_sum = 0
    st = time.time()
    for i in tqdm(range(num_loops)):
        optimizer.zero_grad()
        x_out = model.forward(x)
        loss = crit.forward(x_out, y.to("cuda:3"))
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
    ft = time.time()

    print(f"total loss: {loss_sum}")
    print(f"First layer weight grad:\n{model.layers[0].weight.grad}")
    print(f"Elapsed: {ft - st:8f} total | {(ft - st) / num_loops}/epoch")


def run_dp(  # data parallel
    x: _T,
    y: _T,
    model: TestNN,
    crit: nn.CrossEntropyLoss,
    num_loops: int,
    use_ddp: bool = True,
) -> None:
    """
    Data parallalism:
        - 멀티프로세싱 사용: 각 프로세스는 각자 고유한 device를 사용
        - 모델을 각 프로세스별 device에 복사하지만, 데이터는 device당 개별적인 mini-batch를 forward 함
        - 요구되는 batch의 크기가 클 경우나,
            혹은 batch를 보유한 device갯수만큼 쪼개 학습을 가속하는데에 사용

    DDP를 사용해서 모델을 감싸주어야만 loss.backward()시 ring-all-reduce를 수행하여,
        각 프로세스 별로 복사된 모델 파라미터들이 같은 grad를 가지게 됨.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    print("Running DP / DDP pipeline:")
    dist.init_process_group(backend="nccl")  # torchrun을 사용하여 여러 개의 프로세스 시작

    world_size, global_rank, local_world_size, local_rank = get_process_group_info()

    device = torch.device(f"cuda:{local_rank}")  # local_rank당 device 직접 할당
    torch.cuda.set_device(device)
    model.to(device)  # 모델은 프로세스당 고유한 모델을 가지며, 프로세스당 서로 다른 device에 올리게 됨
    # DDP로 wrapping (이렇게 해야만 loss.backward()시 ring-all-reduce가 작동하여 grad가 싱크됨)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # data parallelism (global rank당 다른 batch를 연산)
    x, y = get_batch_split(x, y, n_chunks=world_size, i=global_rank)
    dist.barrier()

    loss_sum = 0
    st = time.time()
    for _ in tqdm(range(num_loops)):
        optimizer.zero_grad()
        x_out = model.forward(x.to(device))  # 반드시 DDP객체의 .forward()를 사용
        loss = crit.forward(x_out, y.to(device))
        loss.backward()  # <- Ring-all-reduce로 grad를 rank별로 합산 후 rank 수만큼 평균
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # <- 정확한 로깅을 위해 loss도 rank별로 합산 후 평균
        loss_sum += loss.item()
        optimizer.step()
    ft = time.time()

    barrier_delayed(global_rank)
    print(f"total loss: {loss_sum}")
    net = model.module if use_ddp else model
    print(f"First layer weight grad:\n{net.layers[0].weight.grad}")
    print(f"Elapsed: {ft - st:8f} total | {(ft - st) / num_loops:8f}/epoch")
    dist.barrier()

    dist.destroy_process_group()


def run_fsdp(  # fully-sharded data parallel
    x: _T,
    y: _T,
    model: TestNN,
    crit: nn.CrossEntropyLoss,
    num_loops: int,
) -> None:
    import functools

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.nn.utils import parameters_to_vector

    print("Running FSDP pipeline:")
    dist.init_process_group(backend="nccl")
    world_size, global_rank, local_world_size, local_rank = get_process_group_info()

    # auto_wrap_policy를 이용하여 모델을 분산
    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        # DDP처럼 model.to()를 미리 해주고 감싸는 것이 아닌, 여기서 한꺼번에 처리
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # data parallelism (global rank당 다른 batch를 연산)
    x, y = get_batch_split(x, y, n_chunks=world_size, i=global_rank)
    dist.barrier()

    loss_sum = 0
    st = time.time()
    for _ in tqdm(range(num_loops)):
        optimizer.zero_grad()
        x_out = model.forward(x.to(device))
        loss = crit.forward(x_out, y.to(device))
        loss.backward()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss_sum += loss.item()
        optimizer.step()
    ft = time.time()

    barrier_delayed(global_rank)
    print(f"total loss: {loss_sum}")

    barrier_delayed(global_rank)
    # 마법같은 로직으로 쪼개진 FSDP인스턴스 모델의 파라미터들을 모으는 과정
    local_grads = []
    for param in model.parameters():
        if param.grad is not None:
            local_grads.append(param.grad.clone())
    all_grads = parameters_to_vector([grad.flatten() for grad in local_grads])
    print(all_grads)
    dist.barrier()

    print(f"Elapsed: {ft - st:8f} total | {(ft - st) / num_loops:8f}/epoch")
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()
    run, num_loops, use_ddp = args.run_type, args.num_loops, args.ddp

    x, y, model = load_init(args.parent_dir)
    crit = nn.CrossEntropyLoss()

    if run == "basic":
        run_basic(x, y, model, crit, num_loops)
    elif run == "ga":
        run_ga(x, y, model, crit, num_loops)
    elif run == "mp":
        run_mp(x, y, model, crit, num_loops)
    elif run == "dp":
        run_dp(x, y, model, crit, num_loops, use_ddp=use_ddp)
    elif run == "fsdp":
        run_fsdp(x, y, model, crit, num_loops)

    time.sleep(30)

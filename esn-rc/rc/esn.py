from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class EchoStateNetwork:
    def __init__(
        self,
        num_nodes: int,
        weight_connection_prob: float,
        rho: float,
        increment: float,
        bias: float,
        read_in_mul: float,
    ):
        self.num_nodes = num_nodes
        self.sigma = 0.5
        self.weight_connection_prob = weight_connection_prob
        self.rho = rho
        self.increment = increment
        self.bias = bias
        self.read_in_mul = read_in_mul

        self.weights = self._create_weights(
            n=self.num_nodes,
            p=self.weight_connection_prob,
            rho=self.rho,
        )
        self._init_state()
        self._set_bias()

    def _create_weights(self, n: int, p: float, rho: float) -> np.ndarray:
        """RNN의 N x N weights를 초기화"""
        # p 보다 작은 확률로 연결
        weights = np.random.rand(n, n) < p
        # symmetric 연결로 만든 뒤
        weights = (2 * np.random.rand(n, n) - 1) * np.triu(weights, 1)
        weights = weights + np.transpose(weights)
        # max_eigenvalue가 1이 넘지 않도록 normalizing
        max_eigen = np.max(np.abs(np.linalg.eigvals(weights)))
        self.weights = weights / max_eigen * rho

    def _init_state(self) -> None:
        self.state = self.sigma * (-1 + 2 * np.random.rand(self.num_nodes))

    def _set_bias(self) -> None:
        self.bias = self.bias * np.ones(self.num_nodes)
      
    def _update_state(self, input_signal: Optional[np.ndarray] = None):
        """RNN 형태의 점화식을 계산"""
        if input_signal is None:  # initializing input record
            self.state = (
                (1 - self.increment) * self.state
                + self.increment * np.tanh(
                    self.state @ self.weights + self.bias
                )
            )
        else:
            self.state = (
                (1 - self.increment) * self.state
                + self.increment * np.tanh(
                    self.state @ self.weights + self.bias + self.read_in_mul * input_signal,
                )
            )

    def _get_read_in_weights(self, input_size: int) -> np.ndarray:
        return self.sigma * (-1 + 2 * np.random.rand(input_size, self.num_nodes))

    def record_state(self, pre_run: int, init_run: int, input_data: np.ndarray) -> np.ndarray:
        """input에 대한 ESN state의 히스토리(reservoir의 반응)를 모으는 부분"""
        assert init_run < len(input_data)
        t_fin = len(input_data)
        input_size = input_data.shape[-1]
        weights_in = self._get_read_in_weights(input_size)

        state_records = np.zeros((t_fin - init_run, self.num_nodes))
        for _ in range(pre_run):  # starting reservoir
            self._update_state()
        for t in range(init_run):  # injecting input
            self._update_state(feedback=input_data[t] @ weights_in)
        for t in range(init_run, t_fin):
            state_records[t - init_run] = self.state  # recording state
            self._update_state(feedback=input_data[t] @ weights_in)

        return state_records, weights_in

    def predict(
        self,
        steps: int,
        weights_in: np.ndarray,
        read_out: Union[np.ndarray, nn.Module],
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> np.ndarray:
        """self.record_state()에 쓰인 weights_in과, train된 read_out을 사용하여 시계열 예측"""
        # read-out이, 바로 Phi() 매핑
        out = np.zeros((steps, weights_in.shape[0])).astype(float)
        # using pseudo-inverse read_out
        if isinstance(read_out, np.ndarray):
            for t in range(steps):
                out[t] = self.state @ read_out
                # W_in * Phi * s(t)를 input_signal로 주입
                self._update_state(input_signal=out[t] @ weights_in)
        # using nn read_out
        elif isinstance(read_out, nn.Module):
            with torch.no_grad():
                for t in range(steps):
                    out[t] = read_out.forward(
                        torch.from_numpy(self.state).float().unsqueeze(dim=0).to(device)
                    ).cpu().numpy()
                    # W_in * read_out(s(t))를 input_signal로 주입
                    self._update_state(input_signal=out[t] @ weights_in)
        return out

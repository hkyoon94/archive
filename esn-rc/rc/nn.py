import torch
import torch.nn as nn


class ReadOutNN(nn.Module):
    def __init__(self, input_dim: int, inner_d: int, drop_p: float, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, inner_d),
            nn.GELU(),
            nn.BatchNorm1d(inner_d),
            nn.Linear(inner_d, inner_d),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(inner_d, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def train_target(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        epochs: int,
        lr: float = 0.01,
    ) -> float:
        self.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(epochs):
            output = self.forward(input)
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch} loss: {loss.item():.6f}")

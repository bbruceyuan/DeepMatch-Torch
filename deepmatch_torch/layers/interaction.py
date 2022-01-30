
import torch
import torch.nn as nn

# FM 二阶交互
class FactorizationMachines(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X):
        # shape is: (batch_size, num_of_field, embedding_size)
        square_of_sum = torch.pow(
            torch.sum(X, dim=1,),
            2
        )
        sum_of_square = torch.sum(
            torch.pow(
                X, 2
            ),
            dim=1,
        ) # shape is (batch, embeding_size)
        # return shape is (batch, 1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

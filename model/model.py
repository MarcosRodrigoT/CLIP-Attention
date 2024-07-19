import random
import numpy as np
import torch
import torch.nn as nn


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Adapter_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, M, mask):
        # M: Batch of frames embeddings' Matrices -> (B, E, F)
        # mask -> (B, F)
        B, E, F = M.shape
        M = M.permute(0, 2, 1).reshape(B * F, E)  # (B*F, E)

        # Apply MLP to each frame's embedding
        att_scores = self.mlp(M)  # (B*F, 1)
        att_scores = att_scores.reshape(B, F, 1)  # (B, F, 1)

        # Apply mask to attention scores
        att_scores = att_scores * mask.unsqueeze(-1)
        return att_scores


class Adapter_Conv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    # TODO: I leave this here just as a remainder
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, M, mask):
        # M: Batch of frames embeddings' Matrices -> (B, E, F)
        # mask -> (B, F)
        att_scores = self.convs(M)  # (B, 1, F)
        att_scores = att_scores.permute(0, 2, 1)  # (B, F, 1)

        # Apply mask to attention scores
        att_scores = att_scores * mask.unsqueeze(-1)
        return att_scores


class Adapter_Conv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, M, mask):
        # M: Batch of frames embeddings' Matrices -> (B, E, F)
        # mask -> (B, F)
        M = M.unsqueeze(1)  # Add a channel dimension -> (B, 1, E, F)

        # Apply 2D convolutions
        att_scores = self.conv2d_layers(M)  # (B, 1, E, F)

        # Collapse the embedding dimension
        att_scores = att_scores.mean(dim=2)  # (B, 1, F)
        att_scores = att_scores.permute(0, 2, 1)  # (B, F, 1)

        # Apply mask to attention scores
        att_scores = att_scores * mask.unsqueeze(-1)
        return att_scores


class Adapter_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, activation="relu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, M, mask):
        # M: Batch of frames embeddings' Matrices -> (B, E, F)
        # mask -> (B, F)
        M = M.permute(0, 2, 1)  # (B, F, E) as required by the transformer with batch_first=True

        # Apply the transformer encoder
        M_transformed = self.transformer_encoder(M)  # (B, F, E)

        # Apply the final fully connected layers to compute attention scores
        att_scores = self.fc(M_transformed)  # (B, F, 1)

        # Apply mask to attention scores
        att_scores = att_scores * mask.unsqueeze(-1)
        return att_scores


if __name__ == "__main__":
    set_seeds(42)  # Set seeds for reproducibility

    # Example batch with 3 videos, each having different numbers of frames
    video_lengths = [100, 80, 60]
    max_length = max(video_lengths)
    batch_size = len(video_lengths)

    # Initialize matrix of frame embeddings and the mask with zeros
    M = torch.zeros((batch_size, 512, max_length))  # (B, E, F)
    mask = torch.zeros((batch_size, max_length))  # (B, F)

    # Fill matrix of frame embeddings with data and update mask
    for i, length in enumerate(video_lengths):
        M[i, :, :length] = torch.randn((512, length))  # Fill in the valid frames with random data
        mask[i, :length] = 1  # Set the valid frames mask to 1

    # Initialize the network
    adapt_net = Adapter_Conv1D()
    adapt_net.eval()

    # Forward pass
    att_scores = adapt_net(M, mask)

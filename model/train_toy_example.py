import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Adapter


def pad_videos(videos):
    max_len = max(video.shape[1] for video in videos)
    padded_videos = []
    masks = []

    for video in videos:
        padding = max_len - video.shape[1]
        padded_video = torch.nn.functional.pad(video, (0, padding))
        mask = torch.cat([torch.ones(video.shape[1]), torch.zeros(padding)])

        padded_videos.append(padded_video)
        masks.append(mask)

    padded_videos = torch.stack(padded_videos)
    masks = torch.stack(masks)

    return padded_videos, masks


def train(videos, text_descriptions, epochs=100, lr=0.001, lambda_reg=0.1):
    padded_videos, masks = pad_videos(videos)  # (B, E, F), (B, F)
    text_descriptions = torch.stack(text_descriptions)  # (B, E)

    # Initialize the network
    adapt_net = Adapter()
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(adapt_net.parameters(), lr=lr)

    for epoch in range(epochs):
        adapt_net.train()

        # Forward pass
        att_scores = adapt_net(padded_videos, masks)  # (B, F, 1)

        # Compute Out = M @ Att
        Out = torch.matmul(padded_videos, att_scores).squeeze(-1)  # shape (B, E)

        # Compute L2-norm loss
        loss = criterion(Out, text_descriptions)  # (B, E)
        loss = loss.mean(dim=1)  # Reduce over features dimension (B)
        loss = (loss * masks[:, 0]).mean()  # Mask out padded frames (scalar)

        # Add regularization term (sum of attention scores)
        reg_term = lambda_reg * torch.sum(att_scores.squeeze(-1))
        total_loss = loss + reg_term

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss.item():.4f}")

    return adapt_net


if __name__ == "__main__":
    # Example usage
    num_examples = 3
    videos = [torch.randn(512, np.random.randint(10, 100)) for _ in range(num_examples)]  # Example videos with varying lengths
    text_descriptions = [torch.randn(512) for _ in range(num_examples)]  # Example text descriptions

    adapt_net = train(videos, text_descriptions)

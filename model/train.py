import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Adapter_MLP, Adapter_Conv1D, Adapter_Conv2D, Adapter_Transformer


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EmbeddingsDataset(Dataset):
    def __init__(self, video_dir, global_dir):
        self.video_dir = video_dir
        self.global_dir = global_dir
        self.video_files = sorted(os.listdir(video_dir))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        global_path = os.path.join(self.global_dir, video_file)

        video_embeddings = torch.load(video_path).float()  # Convert to float32
        text_embeddings = torch.load(global_path).float()  # Convert to float32

        return video_embeddings, text_embeddings.squeeze(0)


def pad_videos(videos):
    max_len = max(video.shape[0] for video in videos)
    padded_videos = []
    masks = []

    for video in videos:
        padding = max_len - video.shape[0]
        padded_video = torch.nn.functional.pad(video, (0, 0, 0, padding))
        mask = torch.cat([torch.ones(video.shape[0]), torch.zeros(padding)])

        padded_videos.append(padded_video)
        masks.append(mask)

    padded_videos = torch.stack(padded_videos)
    masks = torch.stack(masks)

    return padded_videos, masks


def load_model(adapter):
    if adapter == "mlp":
        return Adapter_MLP()
    elif adapter == "conv1d":
        return Adapter_Conv1D()
    elif adapter == "conv2d":
        return Adapter_Conv2D()
    elif adapter == "transformer":
        return Adapter_Transformer()


def load_hyperparameters(adapter):
    if adapter == "mlp":
        batch_size = 200
        epochs = 201
        lr = 1e-5
        lambda_reg_l1 = 0.001
        lambda_reg_l2 = 0.001
    elif adapter == "conv1d":
        batch_size = 128
        epochs = 501
        lr = 5e-4
        lambda_reg_l1 = 0.001
        lambda_reg_l2 = 0.001
    elif adapter == "conv2d":
        batch_size = 8
        epochs = 201
        lr = 5e-4
        lambda_reg_l1 = 0.001
        lambda_reg_l2 = 0.001
    elif adapter == "transformer":
        batch_size = 64
        epochs = 301
        lr = 5e-4
        lambda_reg_l1 = 0.001
        lambda_reg_l2 = 0.001
    return batch_size, epochs, lr, lambda_reg_l1, lambda_reg_l2


def train(dataloader, adapter, device, epochs=100, lr=0.001, lambda_reg_l1=0.1, lambda_reg_l2=0.1):
    # Initialize the network
    adapt_net = load_model(adapter).to(device)
    adapt_net = nn.DataParallel(adapt_net)  # Wrap the model for data parallelism
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.SGD(adapt_net.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        adapt_net.train()
        total_epoch_loss = 0
        l2_epoch_loss = 0
        reg_epoch_loss = 0
        reg_epoch_loss_l1 = 0
        reg_epoch_loss_l2 = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            videos, text_descriptions = batch
            videos = [video.to(device) for video in videos]
            text_descriptions = [text.to(device) for text in text_descriptions]

            padded_videos, masks = pad_videos(videos)  # (B, F, E), (B, F)
            text_descriptions = torch.stack(text_descriptions)  # (B, E)

            # Forward pass
            att_scores = adapt_net(padded_videos.permute(0, 2, 1), masks)  # (B, F, 1)

            # Compute Out = M @ Att
            Out = torch.matmul(padded_videos.permute(0, 2, 1), att_scores).squeeze(-1)  # shape (B, E)

            # Compute L2-norm loss
            loss = criterion(Out, text_descriptions)  # (B, E)
            loss = loss.mean(dim=1)  # Reduce over features dimension (B)
            loss = (loss * masks[:, 0].to(device)).mean()  # Mask out padded frames (scalar)

            # Add L1 and L2 regularization terms
            l1_reg = lambda_reg_l1 * torch.sum(torch.abs(att_scores.squeeze(-1)))
            l2_reg = lambda_reg_l2 * torch.sum(att_scores.squeeze(-1) ** 2)
            reg_term = l1_reg + l2_reg
            total_loss = loss + reg_term

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            l2_epoch_loss += loss.item()
            reg_epoch_loss += reg_term.item()
            reg_epoch_loss_l1 += l1_reg.item()
            reg_epoch_loss_l2 += l2_reg.item()

        if epoch % 10 == 0:
            loss_t = total_epoch_loss / len(dataloader)
            loss_l = l2_epoch_loss / len(dataloader)
            loss_r = reg_epoch_loss / len(dataloader)
            loss_r_l1 = reg_epoch_loss_l1 / len(dataloader)
            loss_r_l2 = reg_epoch_loss_l2 / len(dataloader)
            tqdm.write(
                f"Epoch [{epoch: >3}/{epochs}], Total Loss: {loss_t: >12.5f} --- MSE Loss: {loss_l: >12.5f} --- Reg Loss:"
                f" {loss_r: >12.5f} --- Reg Loss L1: {loss_r_l1: >12.5f} --- Reg Loss L2: {loss_r_l2: >12.5f}"
            )

    # Save the trained model
    model_save_path = f"model/adapter_{adapter}_model.pth"
    torch.save(adapt_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return adapt_net


if __name__ == "__main__":
    set_seeds(42)  # Set seeds for reproducibility

    video_dir = "embeddings/video"
    global_dir = "embeddings/global"
    adapter = "conv2d"  # "mlp" / "conv1d" / "conv2d" / "transformer"

    # Hyperparameters
    batch_size, epochs, lr, lambda_reg_l1, lambda_reg_l2 = load_hyperparameters(adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    dataset = EmbeddingsDataset(video_dir, global_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    # collate_fn is used so that you don't have to manually extract and process each type of data within your training loop
    # WITHOUT vs WITH collate_fn:
    # [(video1, text1), (video2, text2)]
    # [(video1, video2), (text1, text2)]

    # Train the model
    train(dataloader, adapter, device, epochs=epochs, lr=lr, lambda_reg_l1=lambda_reg_l1, lambda_reg_l2=lambda_reg_l2)

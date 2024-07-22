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
        batch_size = 128
        epochs = 201
        lr = 1e-3
        lambda_reg_l1 = 0
        lambda_reg_l2 = 0
        optimizer_ = "adam"  # "sgd" / "adam"
        loss_fn = "cos"  # "mse" / "cos"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = False
    elif adapter == "conv1d":
        batch_size = 128
        epochs = 201
        lr = 1e-3
        lambda_reg_l1 = 0
        lambda_reg_l2 = 0
        optimizer_ = "adam"  # "sgd" / "adam"
        loss_fn = "cos"  # "mse" / "cos"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = False
    elif adapter == "conv2d":
        batch_size = 8
        epochs = 201
        lr = 1e-3
        lambda_reg_l1 = 0
        lambda_reg_l2 = 0
        optimizer_ = "adam"  # "sgd" / "adam"
        loss_fn = "cos"  # "mse" / "cos"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = False
    elif adapter == "transformer":
        batch_size = 64
        epochs = 201
        lr = 1e-3
        lambda_reg_l1 = 0
        lambda_reg_l2 = 0
        optimizer_ = "adam"  # "sgd" / "adam"
        loss_fn = "cos"  # "mse" / "cos"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = False
    return batch_size, epochs, lr, lambda_reg_l1, lambda_reg_l2, optimizer_, loss_fn, device, multi_gpu


def load_criterion(loss_fn):
    if loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "cos":
        return nn.CosineSimilarity()


def load_optimizer(optimizer, lr, model):
    if optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)


def train(
    adapter,
    device,
    batch_size=128,
    epochs=100,
    lr=0.001,
    lambda_reg_l1=0.1,
    lambda_reg_l2=0.1,
    optimizer_="adam",
    loss_fn="mse",
    multi_gpu=False,
):
    # Display training parameters
    print("Training parameters:")
    print(f"- Adapter: {adapter}")
    print(f"- Loss function: {loss_fn}")
    print(f"- Optimizer: {optimizer_}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {lr}")
    print(f"- Lambda reg L1: {lambda_reg_l1}")
    print(f"- Lambda reg L2: {lambda_reg_l2}")
    print(f"- Multi GPU: {multi_gpu}\n")

    # Load the data
    dataset = EmbeddingsDataset(video_dir, global_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    # collate_fn is used so that you don't have to manually extract and process each type of data within your training loop
    # WITHOUT vs WITH collate_fn:
    # [(video1, text1), (video2, text2)]
    # [(video1, video2), (text1, text2)]

    # Initialize the network
    adapt_net = load_model(adapter).to(device)
    if multi_gpu:
        adapt_net = nn.DataParallel(adapt_net)  # Wrap the model for data parallelism
    criterion = load_criterion(loss_fn)
    optimizer = load_optimizer(optimizer_, lr, model=adapt_net)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        adapt_net.train()
        total_epoch_loss = 0
        l2_epoch_loss = 0
        reg_epoch_loss = 0
        reg_epoch_loss_l1 = 0
        reg_epoch_loss_l2 = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            videos, text_descriptions = batch

            # Move data to device
            videos = [video.to(device) for video in videos]
            text_descriptions = [text.to(device) for text in text_descriptions]

            # Pad videos and stack text descriptions
            padded_videos, masks = pad_videos(videos)  # (B, F, E), (B, F)
            text_descriptions = torch.stack(text_descriptions)  # (B, E)

            # Move data to device (needed for DataParallel)
            padded_videos = padded_videos.to(device)
            masks = masks.to(device)
            text_descriptions = text_descriptions.to(device)

            # Forward pass
            att_scores = adapt_net(padded_videos.permute(0, 2, 1), masks)  # (B, F, 1)

            # Compute Out = M @ Att
            Out = torch.matmul(padded_videos.permute(0, 2, 1), att_scores).squeeze(-1)  # (B, E)

            # Normalize Out by dividing it by the total number of non-dummy frames
            Out = Out.permute(1, 0)  # (E, B)
            Out = Out / torch.sum(masks, dim=1)  # (E, B)
            Out = Out.permute(1, 0)  # (B, E)

            if loss_fn == "mse":
                # Compute MSE loss
                loss = criterion(Out, text_descriptions)  # (B, E)
                loss = loss.mean()  # Reduce over features dimension (scalar)
            elif loss_fn == "cos":
                # Compute cosine similarity loss
                loss = 1 - criterion(Out, text_descriptions)  # (B,)
                loss = loss.mean()  # Average over the batch (scalar)

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
                f"Epoch [{epoch: >3}/{epochs}], Total Loss: {loss_t: >12.5f} --- {loss_fn.upper()} Loss: {loss_l: >12.5f} --- Reg"
                f" Loss: {loss_r: >10.5f} --- Reg Loss L1: {loss_r_l1: >10.5f} --- Reg Loss L2: {loss_r_l2: >10.5f}"
            )

    # Save the trained model
    model_save_path = f"model/adapter_{adapter}_model.pth"
    torch.save(adapt_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return adapt_net


if __name__ == "__main__":
    set_seeds(42)  # Set seeds for reproducibility

    adapter = "conv1d"  # "mlp" / "conv1d" / "conv2d" / "transformer"
    video_dir = "embeddings/video"
    global_dir = "embeddings/global"

    # Hyperparameters
    (
        batch_size,
        epochs,
        lr,
        lambda_reg_l1,
        lambda_reg_l2,
        optimizer_,
        loss_fn,
        device,
        multi_gpu,
    ) = load_hyperparameters(adapter)

    # Train the model
    train(
        adapter,
        device,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lambda_reg_l1=lambda_reg_l1,
        lambda_reg_l2=lambda_reg_l2,
        optimizer_=optimizer_,
        loss_fn=loss_fn,
        multi_gpu=multi_gpu,
    )

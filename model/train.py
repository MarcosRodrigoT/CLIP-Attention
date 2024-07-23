import os
import json
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
    def __init__(self, video_dir, global_dir, gt_file):
        self.video_dir = video_dir
        self.global_dir = global_dir
        self.video_files = sorted(os.listdir(video_dir))
        with open(gt_file, "rb") as f:
            self.ground_truth = json.load(f)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        global_path = os.path.join(self.global_dir, video_file)

        video_embeddings = torch.load(video_path).float()  # Convert to float32
        text_embeddings = torch.load(global_path).float()  # Convert to float32
        ground_truth = self.ground_truth[video_file.split(".")[0]]

        # Pad ground truth with zeros or truncate
        num_frames = video_embeddings.shape[0]
        if len(ground_truth) < num_frames:
            ground_truth = ground_truth + [0] * (num_frames - len(ground_truth))
        elif len(ground_truth) > num_frames:
            ground_truth = ground_truth[:num_frames]
        ground_truth = torch.Tensor(ground_truth)

        return video_embeddings, text_embeddings.squeeze(0), ground_truth


def pad_videos(videos, ground_truth):
    max_len = max(video.shape[0] for video in videos)
    padded_videos = []
    padded_ground_truth = []
    masks = []

    for video, gt in zip(videos, ground_truth):
        padding = max_len - video.shape[0]
        padded_video = torch.nn.functional.pad(video, (0, 0, 0, padding))
        padded_gt = torch.nn.functional.pad(gt, (0, padding))
        mask = torch.cat([torch.ones(video.shape[0]), torch.zeros(padding)])

        padded_videos.append(padded_video)
        padded_ground_truth.append(padded_gt)
        masks.append(mask)

    padded_videos = torch.stack(padded_videos)
    padded_ground_truth = torch.stack(padded_ground_truth)
    masks = torch.stack(masks)

    return padded_videos, padded_ground_truth, masks


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
        lambda_reg_order = 0.01
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
        lambda_reg_order = 0.01
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
        lambda_reg_order = 0.01
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
        lambda_reg_order = 0.01
        optimizer_ = "adam"  # "sgd" / "adam"
        loss_fn = "cos"  # "mse" / "cos"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = False
    return batch_size, epochs, lr, lambda_reg_l1, lambda_reg_l2, lambda_reg_order, optimizer_, loss_fn, device, multi_gpu


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
    lambda_reg_order=0.1,
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
    print(f"- Lambda reg order: {lambda_reg_order}")
    print(f"- Multi GPU: {multi_gpu}\n")

    # Load the data
    dataset = EmbeddingsDataset(video_dir, global_dir, gt_file)
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
        reg_epoch_loss_order = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            videos, text_descriptions, ground_truth = batch

            # Move data to device
            videos = [video.to(device) for video in videos]
            text_descriptions = [text.to(device) for text in text_descriptions]
            ground_truth = [gt.to(device) for gt in ground_truth]

            # Pad videos and stack text descriptions
            padded_videos, padded_ground_truth, masks = pad_videos(videos, ground_truth)  # (B, F, E) / (B, F) / (B, F)
            text_descriptions = torch.stack(text_descriptions)  # (B, E)

            # Move data to device (needed for DataParallel)
            padded_videos = padded_videos.to(device)
            padded_ground_truth = padded_ground_truth.to(device)
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

            # Compute main loss
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

            # Add regularization loss based on GT and Att order
            # Sort Att
            sorted_att_scores, sorted_indices = torch.sort(att_scores.squeeze(-1), descending=True)  # (B, F) / (B, F)
            # Create dummy GT with the number of HL frames in the videos
            batch_size, num_frames = padded_ground_truth.shape  # (B) / (F)  -> The last batch may have a smaller B
            num_ones = torch.sum(padded_ground_truth, dim=1)  # (B)
            mask_ones = torch.arange(num_frames).to(device).expand(batch_size, num_frames) < num_ones.unsqueeze(1)  # (B, F)
            dummy_GT = torch.zeros_like(padded_ground_truth)  # (B, F)
            dummy_GT[mask_ones] = 1.0  # (B, F)
            # Compute regularization loss
            order_reg = lambda_reg_order * nn.MSELoss()(sorted_att_scores, dummy_GT)  # (scalar)

            # Compute total losses
            total_loss = loss + reg_term + order_reg

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            l2_epoch_loss += loss.item()
            reg_epoch_loss += reg_term.item()
            reg_epoch_loss_l1 += l1_reg.item()
            reg_epoch_loss_l2 += l2_reg.item()
            reg_epoch_loss_order += order_reg.item()

        if epoch % 10 == 0:
            loss_t = total_epoch_loss / len(dataloader)
            loss_l = l2_epoch_loss / len(dataloader)
            loss_r = reg_epoch_loss / len(dataloader)
            loss_r_l1 = reg_epoch_loss_l1 / len(dataloader)
            loss_r_l2 = reg_epoch_loss_l2 / len(dataloader)
            loss_r_order = reg_epoch_loss_order / len(dataloader)
            tqdm.write(
                f"Epoch [{epoch: >3}/{epochs}], "
                f"Total Loss: {loss_t: >7.4f} "
                f"--- {loss_fn.upper()} Loss: {loss_l: >7.4f} "
                f"--- Reg Loss: {loss_r: >7.4f} "
                f"--- Reg Loss L1: {loss_r_l1: >7.4f} "
                f"--- Reg Loss L2: {loss_r_l2: >7.4f} "
                f"--- Reg Order: {loss_r_order: >7.4f}"
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
    gt_file = "wikihowto_annt.json"

    # Hyperparameters
    (
        batch_size,
        epochs,
        lr,
        lambda_reg_l1,
        lambda_reg_l2,
        lambda_reg_order,
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
        lambda_reg_order=lambda_reg_order,
        optimizer_=optimizer_,
        loss_fn=loss_fn,
        multi_gpu=multi_gpu,
    )

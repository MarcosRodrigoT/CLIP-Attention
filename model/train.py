import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Adapter


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


def train(dataloader, device, epochs=100, lr=0.001, lambda_reg=0.1):
    # Initialize the network
    adapt_net = Adapter().to(device)
    adapt_net = nn.DataParallel(adapt_net)  # Wrap the model for data parallelism
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(adapt_net.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        adapt_net.train()
        total_epoch_loss = 0
        l2_epoch_loss = 0
        reg_epoch_loss = 0

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

            # Add regularization term (sum of attention scores)
            reg_term = lambda_reg * torch.sum(att_scores.squeeze(-1))
            total_loss = loss + reg_term

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            l2_epoch_loss += loss.item()
            reg_epoch_loss += reg_term.item()

        if epoch % 10 == 0:
            loss_t = total_epoch_loss / len(dataloader)
            loss_l = l2_epoch_loss / len(dataloader)
            loss_r = reg_epoch_loss / len(dataloader)
            tqdm.write(f"Epoch [{epoch: >3}/{epochs}], Total Loss: {loss_t:.5f}\t --- L2 Loss: {loss_l:.5f}\t --- Reg Loss: {loss_r:.5f}")

    return adapt_net


if __name__ == "__main__":
    video_dir = "embeddings/video"
    global_dir = "embeddings/global"

    dataset = EmbeddingsDataset(video_dir, global_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    # collate_fn is used so that you don't have to manually extract and process each type of data within your training loop
    # WITHOUT vs WITH collate_fn:
    # [(video1, text1), (video2, text2)]
    # [(video1, video2), (text1, text2)]

    # Train the model
    epochs = 501
    lr = 5e-4
    lambda_reg = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapt_net = train(dataloader, device, epochs=epochs, lr=lr, lambda_reg=lambda_reg)

    # Save the trained model
    model_save_path = "model/adapter_model.pth"
    torch.save(adapt_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

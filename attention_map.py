import torch
import torchvision
import clip
import numpy as np
import matplotlib.pyplot as plt


def load_video(filepath):
    frames, audio, metadata = torchvision.io.read_video(filepath, start_pts=0, pts_unit="sec")
    return frames, audio, metadata


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def extract_embeddings(frame, model, preprocess, device):
    # [H, W, C] -> [C, H, W]
    image = frame.permute(2, 0, 1)
    image = torchvision.transforms.ToPILImage()(image)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features


def cross_similarity(embeddings):
    # Convert list of tensors to a torch.Tensor
    embeddings = torch.cat(embeddings, axis=0)  # [Batch, Features]
    # Normalize the embeddings to have unit length
    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # [B, F]
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())  # [B, B]

    return similarity_matrix


def plot_similarity(similarity_matrix):
    # Convert the tensor to a NumPy array
    similarity_matrix = similarity_matrix.cpu().numpy()
    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    # Add colorbar to indicate the scale
    plt.colorbar(label="Cosine Similarity")
    # Label axes
    plt.xlabel("Frame")
    plt.ylabel("Frame")
    # Set the ticks to match the frames
    num_frames = similarity_matrix.shape[0]
    plt.xticks(np.arange(0, num_frames, step=max(1, num_frames // 10)))
    plt.yticks(np.arange(0, num_frames, step=max(1, num_frames // 10)))
    # Set axis limits
    plt.xlim(-0.5, num_frames - 0.5)
    plt.ylim(num_frames - 0.5, -0.5)
    # Set axis labels
    plt.title("Cosine Similarity Matrix")
    # Save the plot
    plt.tight_layout()
    plt.savefig("Cosine_sim.png")


def main():
    VIDEO = "wikihow_val/Act-on-a-Movie-Date.mp4"
    TRANSCRIPT = "wikihow_val_transcripts/Act-on-a-Movie-Date.vtt"

    # Load CLIP and its preprocess
    model, preprocess, device = load_model()

    # Load video
    frames, audio, metadata = load_video(VIDEO)

    # Extract info from the video
    video_fps = metadata["video_fps"]
    audio_fps = metadata["audio_fps"]
    num_frames, height, width, channels = frames.shape

    # Extract CLIP embeddings from each frame
    embeddings = []
    for i, frame in enumerate(frames):
        print(f"Processing frame {i}/{len(frames) - 1}", end="\r")
        embeddings.append(extract_embeddings(frame, model, preprocess, device))

    # Create a 2D cross-similarity matrix
    similarity_matrix = cross_similarity(embeddings)

    # Convert the 2D matrix to an image
    plot_similarity(similarity_matrix)


if __name__ == "__main__":
    main()

import os
import cv2
import torch
import torchvision
import clip
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def load_video(filepath):
    frames, audio, metadata = torchvision.io.read_video(filepath, start_pts=0, pts_unit="sec")
    return frames, audio, metadata


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def extract_frame_embeddings(frame, model, preprocess, device):
    # [H, W, C] -> [C, H, W]
    image = frame.permute(2, 0, 1)
    image = torchvision.transforms.ToPILImage()(image)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features


def extract_text_embeddings(text, model, device):
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features


def cross_similarity(frame_embeddings, text_embeddings):
    # Convert list of tensors to a torch.Tensor
    frame_embeddings = torch.cat(frame_embeddings, axis=0)  # [Num_images (N), Features(F)]
    # Normalize the embeddings to have unit length
    normalized_text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)  # [M, F]
    normalized_frame_embeddings = torch.nn.functional.normalize(frame_embeddings, dim=1)  # [N, F]
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(normalized_text_embeddings, normalized_frame_embeddings.t())  # [M, N]

    return similarity_matrix


def plot_similarity(video, description, similarity_matrix, ground_truth):
    num_frames = similarity_matrix.shape[1]

    # Pad ground truth with zeros or truncate
    if len(ground_truth) < num_frames:
        ground_truth = ground_truth + [0] * (num_frames - len(ground_truth))
    elif len(ground_truth) > num_frames:
        ground_truth = ground_truth[:num_frames]

    # Convert the tensor to a NumPy array
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Convert ground truth to rectangle coordinates
    rectangles = []
    start_idx = None
    for idx, value in enumerate(ground_truth):
        if value == 1 and start_idx is None:
            start_idx = idx
        elif value == 0 and start_idx is not None:
            rectangles.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        rectangles.append((start_idx, num_frames - 1))

    # Create the combined plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot without ground truth
    ax = axes[0]
    cax = ax.imshow(similarity_matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_title("Cosine Similarity Matrix")
    ax.set_xlabel("Frame")
    ax.set_xticks(np.arange(0, num_frames, step=max(1, num_frames // 10)))
    ax.set_yticks([0])
    ax.set_yticklabels([description], rotation=90, va="center")
    fig.colorbar(cax, ax=ax, label="Cosine Similarity")

    # Plot with ground truth
    ax = axes[1]
    cax = ax.imshow(similarity_matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_title("Ground Truth")
    ax.set_xlabel("Frame")
    ax.set_xticks(np.arange(0, num_frames, step=max(1, num_frames // 10)))
    ax.set_yticks([0])
    ax.set_yticklabels([description], rotation=90, va="center")
    fig.colorbar(cax, ax=ax, label="Cosine Similarity")
    # Draw rectangles for ground truth
    for start, end in rectangles:
        rect = patches.Rectangle(
            (start - 0.5, -0.5),
            end - start + 1,
            1,
            linewidth=1,
            edgecolor="yellow",
            facecolor="yellow",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(rect)

    # Set title
    plt.suptitle(f"Video: {video.split('/')[-1]}")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"results/frames_global/{video.split('/')[-1].split('.')[0]}.png")

    return fig, axes


def create_video_and_plot(video_path, description, similarity_matrix, ground_truth):
    # Scaling factor for resolution
    scale_factor = 4

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Plot similarity matrix and rectangles
    fig, axes = plot_similarity(video_path, description, similarity_matrix, ground_truth)
    ax1, ax2 = axes

    # Create vertical lines to move along the frames
    line1 = ax1.axvline(x=0, color="green", linestyle="-")
    line2 = ax2.axvline(x=0, color="green", linestyle="-")

    def update(frame):
        # Update the vertical lines
        line1.set_xdata([frame])
        line2.set_xdata([frame])
        fig.canvas.draw()
        return [line1, line2]

    # Create VideoWriter object to save the output video with increased resolution
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_width = 2 * width * scale_factor
    out_height = height * scale_factor
    out = cv2.VideoWriter(f"results/frames_global/{video_path.split('/')[-1].split('.')[0]}.avi", fourcc, fps, (out_width, out_height))

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Update the animation
        update(frame_idx)

        # Convert the plot to an image
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

        # Scale the video frame and the plot image
        frame = cv2.resize(frame, (width * scale_factor, height * scale_factor))
        plot_img = cv2.resize(plot_img, (width * scale_factor, height * scale_factor))

        # Combine the video frame and the plot image
        combined_img = np.hstack((frame, plot_img))

        # Write the frame to the output video
        out.write(combined_img)

    cap.release()
    out.release()


def main(CREATE_VIDEO, METHOD):
    for i, video in enumerate(sorted(os.listdir("wikihow_val"))):
        print(f"Processing video {i}/{len(os.listdir('wikihow_val')) - 1}")

        VIDEO = os.path.join("wikihow_val", video)

        with open("wikihowto_annt.json", "r") as f:
            GROUND_TRUTH = json.load(f)[f"{video.split('.')[0]}"]

        # Skip this iteration if there is no ground truth for this video
        if not GROUND_TRUTH:
            print(f"Skipping video {video}")
            continue

        if METHOD == "description":
            global_description_file = f"transcripts/global/{video.split('.')[0]}.vtt"
            if os.path.exists(global_description_file):
                with open(global_description_file, "r") as file:
                    GLOBAL_DESCRIPTION = file.readline()
            else:
                GLOBAL_DESCRIPTION = " ".join(video.split(".")[0].split("-"))

        elif METHOD == "filename":
            GLOBAL_DESCRIPTION = " ".join(video.split(".")[0].split("-"))

        # Load CLIP and its preprocess
        model, preprocess, device = load_model()

        # Load video
        frames, audio, metadata = load_video(VIDEO)

        # Extract info from the video
        video_fps = metadata["video_fps"]
        audio_fps = metadata["audio_fps"]
        num_frames, height, width, channels = frames.shape

        # Extract CLIP embeddings from each frame
        frame_embeddings = []
        for frame in frames:
            frame_embeddings.append(extract_frame_embeddings(frame, model, preprocess, device))

        # Extract CLIP embeddings from the global description
        text_embeddings = extract_text_embeddings(GLOBAL_DESCRIPTION, model, device)

        # Create a 2D cross-similarity matrix
        similarity_matrix = cross_similarity(frame_embeddings, text_embeddings)

        if CREATE_VIDEO:
            # Create animated video and plots
            create_video_and_plot(VIDEO, GLOBAL_DESCRIPTION, similarity_matrix, GROUND_TRUTH)
        else:
            # Convert the 2D matrix to an image
            plot_similarity(VIDEO, GLOBAL_DESCRIPTION, similarity_matrix, GROUND_TRUTH)


if __name__ == "__main__":
    CREATE_VIDEO = False
    METHOD = "filename"  # "filename" or "description"

    main(CREATE_VIDEO, METHOD)

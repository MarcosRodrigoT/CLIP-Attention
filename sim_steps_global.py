import os
import re
import torch
import torchvision
import clip
import json
import textwrap
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def clean_cvpr_like_transcripts(steps):
    # If the steps consist of a single line, it's probably a corrupt file
    if len(steps) == 1:
        return []

    cleaned_steps = []
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")

    for step in steps:
        step = step.strip()
        if not step or timestamp_pattern.match(step):
            continue
        if timestamp_pattern.match(step[:29]):
            step = step[30:]
        if len(step) > 77:
            step = step[:77]
        cleaned_steps.append(step)

    return cleaned_steps


def load_video(filepath):
    frames, audio, metadata = torchvision.io.read_video(filepath, start_pts=0, pts_unit="sec")
    return frames, audio, metadata


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def extract_text_embeddings(text, model, device):
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features


def cross_similarity(global_embeddings, steps_embeddings):
    # Normalize the embeddings to have unit length
    normalized_global_embeddings = torch.nn.functional.normalize(global_embeddings, dim=1)  # [1, F]
    normalized_steps_embeddings = torch.nn.functional.normalize(steps_embeddings, dim=1)  # [M, F]
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(normalized_steps_embeddings, normalized_global_embeddings.t())  # [M, 1]

    return similarity_matrix


def plot_similarity(video, description, steps, similarity_matrix, method):
    num_steps = similarity_matrix.shape[0]

    # Convert the tensor to a NumPy array
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Create the combined plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Handle steps description to fit the plot
    wrapped_steps = [textwrap.fill(step, 40) for step in steps]

    # Plot without ground truth
    cax = ax.imshow(similarity_matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_title("Cosine Similarity Matrix")
    ax.set_xticks([0])
    ax.set_yticks(np.arange(num_steps))
    ax.set_xticklabels([description], rotation=0, va="center", ha="center")
    ax.set_yticklabels(wrapped_steps, rotation=0, ha="right")
    fig.colorbar(cax, ax=ax, label="Cosine Similarity")

    # Set title
    plt.suptitle(f"Video: {video.split('/')[-1]}")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"results/steps_global/{method}/{video.split('/')[-1].split('.')[0]}.png")


def main(METHOD):
    for i, video in enumerate(sorted(os.listdir("wikihow_val"))):
        print(f"Processing video {i}/{len(os.listdir('wikihow_val')) - 1} -> {video}")

        VIDEO = os.path.join("wikihow_val", video)

        with open("wikihowto_annt.json", "r") as f:
            GROUND_TRUTH = json.load(f)[f"{VIDEO.split('/')[-1].split('.')[0]}"]

        # Skip this iteration if there is no ground truth for this video
        if not GROUND_TRUTH:
            print(f"Skipping video {video}")
            continue

        GLOBAL_DESCRIPTION = " ".join(video.split(".")[0].split("-"))

        if METHOD == "from_transcript":
            steps_file = f"transcripts/steps/generic_from_transcript/{video.split('.')[0]}.vtt"
            if not os.path.exists(steps_file):
                print(f"Skipping video {video}")
                continue
            with open(steps_file, "r") as file:
                steps_ = file.readlines()
            # Remove elements that do not start with a numeration like "1. ", "2. ", etc.
            filtered_steps = [step for step in steps_ if re.match(r"^\d+\.\s", step.strip())]
            # In the case the transcript is corrupt skip
            if not filtered_steps:
                print(f"Skipping video {video}")
                continue
            # Remove numeration from each step
            steps = [step.split(". ", 1)[1] for step in filtered_steps]

        elif METHOD == "from_filename":
            steps_file = f"transcripts/steps/generic_from_filename/{video.split('.')[0]}.vtt"
            if not os.path.exists(steps_file):
                print(f"Skipping video {video}")
                continue
            with open(steps_file, "r") as file:
                steps_ = file.readlines()
            # Remove elements that do not start with a numeration like "1. ", "2. ", etc.
            filtered_steps = [step for step in steps_ if re.match(r"^\d+\.\s", step.strip())]
            # In the case the transcript is corrupt skip
            if not filtered_steps:
                print(f"Skipping video {video}")
                continue
            # Remove numeration from each step
            steps = [step.split(". ", 1)[1] for step in filtered_steps]

        elif METHOD == "cvpr_paper_like":
            steps_file = f"transcripts/steps/cvpr_paper_like/{video.split('.')[0]}.vtt"
            if not os.path.exists(steps_file):
                print(f"Skipping video {video}")
                continue
            with open(steps_file, "r") as file:
                steps_ = file.readlines()
            # Clean transcripts
            steps = clean_cvpr_like_transcripts(steps_)
            # In the case the transcript is corrupt skip
            if not steps:
                print(f"Skipping video {video}")
                continue

        # Load CLIP and its preprocess
        model, preprocess, device = load_model()

        # Extract CLIP embeddings from the steps' sentences and global description
        global_embeddings = extract_text_embeddings(GLOBAL_DESCRIPTION, model, device)
        steps_embeddings = extract_text_embeddings(steps, model, device)

        # Create a 2D cross-similarity matrix
        similarity_matrix = cross_similarity(global_embeddings, steps_embeddings)

        # Convert the 2D matrix to an image
        plot_similarity(VIDEO, GLOBAL_DESCRIPTION, steps, similarity_matrix, METHOD)


if __name__ == "__main__":
    main(METHOD="from_filename")

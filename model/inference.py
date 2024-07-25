import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from model import Adapter_MLP, Adapter_Conv1D, Adapter_Conv2D, Adapter_Transformer


def load_model(adapter):
    if adapter == "mlp":
        return Adapter_MLP()
    elif adapter == "conv1d":
        return Adapter_Conv1D()
    elif adapter == "conv2d":
        return Adapter_Conv2D()
    elif adapter == "transformer":
        return Adapter_Transformer()


def load_trained_model(adapter, model_path, device):
    model = load_model(adapter).to(device)
    # model = nn.DataParallel(model)  # Wrap the model for data parallelism
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def summarize_video(video_embeddings_path, model, device, summar_len):
    # Load the video embeddings
    video_embeddings = torch.load(video_embeddings_path).float().to(device)  # Convert to float32 and move to device

    # Add batch dimension (1, F, E)
    video_embeddings = video_embeddings.unsqueeze(0).permute(0, 2, 1)

    # Create a mask (all ones since we have no padding)
    mask = torch.ones(video_embeddings.shape[2]).unsqueeze(0).to(device)

    # Get the attention scores from the model
    with torch.no_grad():
        att_scores = model(video_embeddings, mask)  # (1, F, 1)
        att_scores = att_scores.squeeze(0).squeeze(-1)  # (F,)

    # print(f"************ [Min, Max] Attention Scores:  [{torch.min(att_scores)}, {torch.max(att_scores)}]")
    # print(f"************ Sum of Attention Scores:  {torch.sum(att_scores)}")

    # Determine the threshold to convert attention scores to binary values
    num_frames = att_scores.shape[0]
    num_summary_frames = int(num_frames * summar_len)

    # Get the indices of the top attention scores
    _, top_indices = torch.topk(att_scores, num_summary_frames)

    # Initialize the binary attention scores with zeros
    binary_att_scores = torch.zeros_like(att_scores)

    # Set the top attention scores to 1
    binary_att_scores[top_indices] = 1

    return binary_att_scores


def run_inference():
    SUMMAR_LEN = 0.6  # Desired summary length (e.g., 60% of the frames)
    ADAPTER = "conv1d"  # "mlp" / "conv1d" / "conv2d" / "transformer"

    # Load the trained model
    model_path = f"model/adapter_{ADAPTER}_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(ADAPTER, model_path, device)

    videos = sorted(os.listdir("embeddings/video/"))
    precisions = []
    recalls = []
    fscores = []
    for video in tqdm(videos, desc="Processing videos"):
        video_embeddings_path = os.path.join("embeddings/video/", video)

        # Summarize the new video
        att_scores = summarize_video(video_embeddings_path, model, device, SUMMAR_LEN)
        att_scores = att_scores.detach().cpu().numpy()

        # Get the ground truth
        with open("wikihowto_annt.json", "r") as f:
            ground_truth = json.load(f)[f"{video_embeddings_path.split('/')[-1].split('.')[0]}"]

        # Pad ground truth with zeros or truncate
        if len(ground_truth) < len(att_scores):
            ground_truth = ground_truth + [0] * (len(att_scores) - len(ground_truth))
        elif len(ground_truth) > len(att_scores):
            ground_truth = ground_truth[: len(att_scores)]
        ground_truth = np.array(ground_truth)

        # Compute precision, recall, and F1-score
        precision = precision_score(ground_truth, att_scores)
        recall = recall_score(ground_truth, att_scores)
        fscore = f1_score(ground_truth, att_scores)

        # Append metrics for later averaging
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    # Average metrics
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_fscore = np.mean(fscores)

    # Print some metrics
    print("\nInference results on WikiHow dataset:")
    print(f"- Average Precision: {avg_precision * 100:.2f}%")
    print(f"- Average Recall: {avg_recall * 100:.2f}%")
    print(f"- Average F-Score: {avg_fscore * 100:.2f}%")


if __name__ == "__main__":
    run_inference()

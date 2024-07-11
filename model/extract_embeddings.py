import os
import torch
import torchvision
import json
import clip


def load_video(filepath):
    frames, audio, metadata = torchvision.io.read_video(filepath, start_pts=0, pts_unit="sec")
    return frames, audio, metadata


def load_ground_truth():
    with open("wikihowto_annt.json", "r") as f:
        ground_truth = json.load(f)
    return ground_truth


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def extract_frame_embeddings(frames, model, preprocess, device):
    frame_embeddings = []
    for frame in frames:
        image = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        image = torchvision.transforms.ToPILImage()(image)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        frame_embeddings.append(image_features)
    frame_embeddings = torch.cat(frame_embeddings, axis=0)
    return frame_embeddings


def extract_text_embeddings(text, model, device):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


def main():
    VIDEO_FILES = sorted(os.listdir("wikihow_val"))
    GROUND_TRUTH = load_ground_truth()

    for i, video_file in enumerate(VIDEO_FILES):
        print(f"Processing video {i}/{len(VIDEO_FILES) - 1} -> {video_file}")

        video_path = os.path.join("wikihow_val", video_file)
        global_description = " ".join(video_file.split(".")[0].split("-"))
        ground_truth = GROUND_TRUTH[f"{video_file.split('.')[0]}"]

        # Load CLIP and its preprocess
        model, preprocess, device = load_model()

        # Load video
        frames, audio, metadata = load_video(video_path)

        # Extract CLIP embeddings from each frame
        frame_embeddings = extract_frame_embeddings(frames, model, preprocess, device)
        frame_embeddings = frame_embeddings.detach().cpu()

        # Extract CLIP embeddings from the global description
        text_embeddings = extract_text_embeddings(global_description, model, device)
        text_embeddings = text_embeddings.detach().cpu()

        # Save embeddings to disk
        with open(f"embeddings/video/{video_file.split('.')[0]}.pt", "wb") as f:
            torch.save(frame_embeddings, f)
        with open(f"embeddings/global/{video_file.split('.')[0]}.pt", "wb") as f:
            torch.save(text_embeddings, f)


if __name__ == "__main__":
    main()

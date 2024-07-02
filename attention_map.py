import torch
import torchvision
import clip


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


if __name__ == "__main__":
    main()

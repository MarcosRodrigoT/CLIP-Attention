import os
from PIL import Image, ImageDraw, ImageFont


def add_title(image, title):
    # Create a new image with space for the title
    title_height = 50
    new_image = Image.new("RGB", (image.width, image.height + title_height), (255, 255, 255))

    # Draw the title on the new image
    draw = ImageDraw.Draw(new_image)
    try:
        # Load a truetype or opentype font file, and create a font object.
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # If the specified font is not available, use the default font
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    draw.text(((new_image.width - text_width) / 2, (title_height - text_height) / 2), title, fill="black", font=font)

    # Paste the original image below the title
    new_image.paste(image, (0, title_height))

    return new_image


def combine_images(image_paths, image_titles, output_path):
    # Open all images and add titles
    images = [add_title(Image.open(image_path), title) for image_path, title in zip(image_paths, image_titles)]

    # Create a blank image with the combined height
    combined_image = Image.new("RGB", (2000, len(images) * (1000 + 50)))  # 50 is the height of the title area

    # Paste each image into the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the combined image
    combined_image.save(output_path)


def main():
    for i, video in enumerate(sorted(os.listdir("wikihow_val"))):
        print(f"Processing video {i}/{len(os.listdir('wikihow_val')) - 1} -> {video}")

        # Individual images' paths
        frames_frames = f"results/frames_frames/{video.split('.')[0]}.png"
        frames_global = f"results/frames_global/{video.split('.')[0]}.png"
        frames_sentences = f"results/frames_sentences/{video.split('.')[0]}.png"
        frames_steps_f = f"results/frames_steps/from_filename/{video.split('.')[0]}.png"
        frames_steps_t = f"results/frames_steps/from_transcript/{video.split('.')[0]}.png"
        frames_steps_c = f"results/frames_steps/cvpr_paper_like/{video.split('.')[0]}.png"
        steps_global = f"results/steps_global/from_filename/{video.split('.')[0]}.png"
        steps_steps_f = f"results/steps_steps/from_filename/{video.split('.')[0]}.png"
        steps_steps_t = f"results/steps_steps/from_transcript/{video.split('.')[0]}.png"
        steps_steps_c = f"results/steps_steps/cvpr_paper_like/{video.split('.')[0]}.png"

        # Combine the images
        image_paths = [
            frames_frames,
            frames_global,
            # frames_sentences,
            frames_steps_f,
            # frames_steps_t,
            # frames_steps_c,
            steps_global,
            steps_steps_f,
            # steps_steps_t,
            # steps_steps_c,
        ]
        image_titles = [
            "frames_frames",
            "frames_global",
            # "frames_sentences",
            "frames_steps (filename)",
            # "frames_steps (transcript)",
            # "frames_steps (cvpr)",
            "steps_global (filename)",
            "steps_steps (filename)",
            # "steps_steps (transcript)",
            # "steps_steps (cvpr)",
        ]
        output_path = f"results/combined/{video.split('.')[0]}.png"

        # Check if all image files exist
        dont_exist = [file for file in image_paths if not os.path.exists(file)]
        if dont_exist:
            print(f"Skipping video {video} -> These files do not exist: {dont_exist}")
            continue

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Combine images into one
        combine_images(image_paths, image_titles, output_path)


if __name__ == "__main__":
    main()

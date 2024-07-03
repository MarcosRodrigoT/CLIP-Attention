import re
import glob


def clean_transcript_w_timestamps(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    cleaned_lines = []
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}")
    caption_text = ""
    current_start_timestamp = None

    for line in lines:
        line = line.strip()
        if timestamp_pattern.match(line):
            timestamps = line.split(" align:start position:0%")[0].split(" --> ")
            current_start_timestamp, current_end_timestamp = timestamps
        elif line and "<c>" not in line and not line.startswith(("WEBVTT", "Kind:", "Language:")):
            if caption_text and line == caption_text:
                # Update the end time of the last timestamp
                previous_timestamps[-1] = current_end_timestamp
            else:
                if caption_text:
                    cleaned_lines.append(f"{previous_timestamps[0]} --> {previous_timestamps[-1]}\n{caption_text}\n\n")
                caption_text = line
                previous_timestamps = [current_start_timestamp, current_end_timestamp]

    if caption_text:
        cleaned_lines.append(f"{previous_timestamps[0]} --> {previous_timestamps[-1]}\n{caption_text}\n\n")

    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(cleaned_lines)


def clean_transcript_wo_timestamps(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    cleaned_lines = []
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")

    for line in lines:
        line = line.strip()
        if not line or timestamp_pattern.match(line):
            continue
        cleaned_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as file:
        for line in cleaned_lines:
            file.write(line + "\n")


if __name__ == "__main__":
    for original_transcript in glob.glob("wikihow_val_transcripts/*.vtt"):
        clean_transcript_w_timestamps(
            original_transcript,
            f"transcripts/w_timestamps/{original_transcript.split('/')[-1]}",
        )
        clean_transcript_wo_timestamps(
            f"transcripts/w_timestamps/{original_transcript.split('/')[-1]}",
            f"transcripts/wo_timestamps/{original_transcript.split('/')[-1]}",
        )

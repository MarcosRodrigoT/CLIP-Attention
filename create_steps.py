import glob
import requests
import json


def get_response(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama3:70b", "prompt": prompt, "stream": False}  # llama3:8b or llama3:70b

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result["response"].strip()
    else:
        print(f"Error: {response.status_code}")
        return ""


def create_generic_steps(input_file, output_file, mode="transcript"):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if mode == "transcript":
        # If file ends with DELETE, ignore this file
        if lines[-1] == "DELETE":
            print(f"\t- Skipping file: {input_file}")
            return

        # Append all sentences to a list (although there should only be 1 line)
        sentences = []
        for line in lines:
            line = line.strip()
            if line:
                sentences.append(line)

        # Convert the list of sentences into a continuous text
        text = " ".join(sentences)

    elif mode == "filename":
        words = input_file.split("/")[-1].split(".")[0].split("-")
        text = " ".join(words)

    with open(output_file, "w", encoding="utf-8") as file:
        response = get_response(
            "You are now an instructor. Given a topic, you must write a numbered list of the steps usually involved in achieving this"
            " topic. Describe each of these steps very briefly, in only one sentence if possible. Write each step in a new line. Refrain"
            " from adding any additional comments, notes, or remarks in your response, simply return the numbered list. This is because I"
            " will directly copy an use whatever you return me, so please do not add anything else but just the numbered list. Here is the"
            f" topic: {text}"
        )

        file.write(response)


def create_cvpr_like_summary(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Append all sentences to a list
    sentences = []
    for line in lines:
        if line:
            sentences.append(line)

    # Convert the list of sentences into a continuous text
    text = "".join(sentences)

    with open(output_file, "w", encoding="utf-8") as file:
        response = get_response(
            "I am providing you with a transcribed narration from a video, complete with timestamps preceeding each sentence (in the form"
            " of 'start --> end', such as '00:00:02.070 --> 00:00:04.230'). Please generate an extractive summary from this text. Here are"
            " your instructions: 1. The summary should consist of only the most critical and informative moments from the video. 2. Do not"
            " paraphrase or reword the sentences. Maintain their original wording. 3. Each sentence you extract for the summary must"
            " include its original timestamp. 4. It is very important that you stick to these instructions. You must never add any"
            " additional comments, notes, remarks, or even words in your response, simply return the summary. I will directly copy an use"
            " whatever you return me, without even reading any of it, that is why it is very important that you only return the summary"
            f" without adding anything else such as 'here is the summary:'. This is the transcript: {text}"
        )

        # Make sure there is no "Here is the summary:\n\n"
        # Find the position of the first occurrence of ":\n\n"
        delimiter_pos = response.find(":\n\n")

        # If the delimiter is found, remove everything before and including it
        if delimiter_pos != -1:
            response = response[delimiter_pos + 3 :]

        file.write(response)


if __name__ == "__main__":
    for global_description in sorted(glob.glob("transcripts/global/*")):
        print(f"Processing file: {global_description}")

        # Create list of steps based on the transcripts
        output_file = f"transcripts/steps/generic_from_transcript/{global_description.split('/')[-1]}"
        create_generic_steps(global_description, output_file, mode="transcript")

        # Create list of steps based on the file names
        output_file = f"transcripts/steps/generic_from_filename/{global_description.split('/')[-1]}"
        create_generic_steps(global_description, output_file, mode="filename")

        # Create a CVPR-paper-like summary
        transcript_w_timestamps = f"transcripts/w_timestamps/{global_description.split('/')[-1]}"
        output_file = f"transcripts/steps/cvpr_paper_like/{global_description.split('/')[-1]}"
        create_cvpr_like_summary(transcript_w_timestamps, output_file)

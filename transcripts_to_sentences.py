import glob
import requests
import json


def get_full_sentence(prompt):
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


def process_transcript(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Append all sentences to a list
    sentences = []
    for line in lines:
        line = line.strip()
        if line:
            sentences.append(line)

    # Convert the list of sentences into a continuous text
    text = " ".join(sentences)

    with open(output_file, "w", encoding="utf-8") as file:
        full_sentences = get_full_sentence(
            "Separate the following text into full sentences. Write each full sentence in a new line. Do not create a numbered list,"
            " simply output each full sentence in a new line. Refrain from adding any additional comments, notes, or remarks in your"
            " response. You must never add notes in your response, you must be self sufficient and proceed without asking for cofirmation"
            " of any type nor indication on why you are proceeding in a certain way. Simply add punctuation marks where necessary and use"
            " upper/lower case letters where necessary. Also correct individual words that are written incorrectly. If the complete text I"
            " give you is empty or almost empty and it appears to be incomplete or fragmented, you must always return the word 'DELETE' as"
            " a final line. Only do these if the complete text seems to be fragmented or incomplete, never do this if there is at least"
            " one full coherent sentence. In case of doubt return the word 'DELETE', never ask for confirmation from my part. It is very"
            f" important to stick to the instructions given. Here is the text: {text}"
        )

        file.write(full_sentences)


def clean_sentences(file):
    with open(file, "r") as f:
        lines = f.readlines()

    sentences_to_write = []
    for line in lines:
        if line.endswith(":\n"):
            continue

        sentences = line.split(". ")
        sentences = [s + "." if i < len(sentences) - 1 else s for i, s in enumerate(sentences)]

        # Add cleaned sentences to list
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence:
                sentences_to_write.append(clean_sentence)

    with open(file, "w") as f:
        for i, sentence in enumerate(sentences_to_write):
            if i < len(sentences_to_write) - 1:
                f.write(sentence + "\n")
            else:
                f.write(sentence)


if __name__ == "__main__":
    for transcript in sorted(glob.glob("transcripts/wo_timestamps/*")):
        print(f"Processing file: {transcript}")

        output_file = f"transcripts/sentences/{transcript.split('/')[-1]}"
        process_transcript(transcript, output_file)

        clean_sentences(output_file)

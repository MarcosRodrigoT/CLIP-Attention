import glob
import requests
import json


def transcripts_to_delete(file):
    with open(file, "r") as f:
        lines = f.readlines()

    # If file ends with DELETE, delete everythin else
    if lines[-1] == "DELETE":
        print(f"Deleting file: {file}")
        with open(file, "w") as f:
            f.write("DELETE")


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

    # If file ends with DELETE, ignore this file
    if lines[-1] == "DELETE":
        print(f"\t- Skipping file: {input_file}")
        return

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
            "Summarize the following text into a single sentence that describes the topic in general terms. Keep the sentence short but"
            " meaningful. You must response only with this sentence, do not add any other word or sentence in your response, as I will use"
            " this sentence directly and I will not be able to differentiate the sentence summarizing the text from any other text you use"
            " to communicate with me. For instance, if I provide you a long text describing all the steps involved in the preparation of a"
            " dish of pasta, you should return a sentence like 'Recipe for a dish of pasta'. Let me give you another example, if I provide"
            " you with a long text talking about how to apply makeup, you should return something similar to 'Applying makeup'. Here is"
            f" the text: {text}"
        )

        file.write(full_sentences)


if __name__ == "__main__":
    for transcript in sorted(glob.glob("transcripts/sentences/*")):
        print(f"Processing file: {transcript}")

        transcripts_to_delete(transcript)

        output_file = f"transcripts/global/{transcript.split('/')[-1]}"
        process_transcript(transcript, output_file)

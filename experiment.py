from datasets import load_dataset
from tqdm.auto import tqdm

print()

text_data = ""
dataset = load_dataset("bookcorpus")

iter = 0

for sample in tqdm(dataset["train"]):
    text = sample['text']
    text = text.replace(" .", ".")
    text = text.replace(".", ". ")
    text = text.replace("''", '"')
    text = text.replace(" ,", ",")
    text = text.replace("``", '"')
    text = text.replace('" ', '"')
    text = text.replace(" '", "'")
    text = text.replace(' "', '"')
    text = text.replace(" n't", "n't")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    
    if text[0] == '"' and text[-1] == '"':
        new_text = "\n" + text + "\n"
        text_data += new_text
    elif text[0] == '"' and text[-1] != '"':
        new_text = "\n" + text
        text_data += new_text
    elif text[0] != '"' and text[-1] == '"':
        new_text = text + "\n"
        text_data += new_text
    else:
        text_data += text

    iter += 1

    if iter >= 3000000:
        break

with open("book_corpusRAW.txt", "w") as outputFile:
    outputFile.write(str(text_data))
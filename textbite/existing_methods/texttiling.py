import nltk
from nltk.corpus import stopwords


PATH = r"/home/martin/textbite/data/gt/3f9b00b0-681f-11dc-9c9a-000d606f5dc6.txt"
STOPWORDS_PATH = r"/home/martin/textbite/czech"


if __name__ == "__main__":
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = f.read().strip().split("\n")

    with open(PATH, "r") as f:
        text = f.read()

    tokenizer = nltk.tokenize.TextTilingTokenizer(stopwords=stopwords)
    res = tokenizer.tokenize(text)

    for p in res:
        print(f"{p}\n\n\n")

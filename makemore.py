# makes more of the data it has given
from models.bigram.bigram import Bigram
from models.ngram.ngram import Ngram


def clean_data(data):
    clean_data = []
    valid_chars = 'âabcçefgğhıijklmnoöprsştuüvyz'
    for sample in data:
        sample = sample.lower()
        # check if all chars are valid
        if any([s not in valid_chars for s in sample]):
            continue
        clean_data.append(sample)
    return clean_data

def load_data(file_path: str): # reads csv files
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    # get only the first column
    data = [line.split(',')[0] for line in data]
    return clean_data(data)

def main(file_name):
    data = load_data(file_name)
    bigram_model = Bigram(data)
    print(bigram_model.get_next())

    ngram_model = Ngram(data, 3)
    print(ngram_model.generate_next())


if __name__=="__main__":
    main('names.csv')

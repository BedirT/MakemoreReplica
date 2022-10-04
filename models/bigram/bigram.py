# Bigram implementations with count and neural network seperately.

import torch


# DATA SETUP - Shared
def parse_data_old() -> list:
    file_name = 'data/names_old.csv'
    with open(file_name, 'r') as f:
        data = f.read().splitlines()
    return [d for d in data]

def parse_data_new() -> list:
    file_name = 'data/names_new.csv'
    with open(file_name, 'r') as f:
        data = f.read().splitlines()
    return [d.split(',')[0] for d in data]

def setup_data(data) -> None:
    clean_data = []
    valid_chars = 'âabcçdefgğhıiîjklmnoöprsştuüûvyz'
    for name in data:
        name = name.replace('İ', 'i') # Capital İ is two chars in ascii
        if all([c in valid_chars for c in name.lower()]):
            clean_data.append(name.lower())
    new_data = clean_data

    itos = {i+1:c for i, c in enumerate(valid_chars)}
    itos[0] = '.' # start and end token
    stoi = {c:i for i, c in itos.items()}

    return new_data, itos, stoi


class BigramWithTable:

    def __init__(self, seed=-1):
        if seed == -1:
            seed = torch.randint(0, 9999999, (1,)).item()
        
        self.generator = torch.Generator().manual_seed(seed)
        self.data, self.itos, self.stoi = setup_data(parse_data_new())

        # Setup bigram table
        self._setup_table()

    def _setup_table(self) -> None:
        table = torch.ones((len(self.stoi), len(self.stoi)), dtype=int)
        for name in self.data:
            tokenized_name = ['.'] + list(name) + ['.']
            for ch1, ch2 in zip(tokenized_name, tokenized_name[1:]):
                table[self.stoi[ch1], self.stoi[ch2]] += 1

        # Normalize the table to get probabilities
        table = table / table.sum(dim=1, keepdim=True)
        self.table = table

    def _get_next(self, idx:int) -> int:
        ''' Predict the next letter idx from the letter idx before '''
        return torch.multinomial(self.table[idx], 
                                 1, generator=self.generator)

    def sample_name(self) -> str:
        ''' Returns a generated name '''
        cur_idx = 0
        gen_name = ""
        while True:
            # Sample from cur_idx
            cur_idx = self._get_next(cur_idx).item()
            if cur_idx == 0:
                break
            gen_name += self.itos[cur_idx]
        return gen_name


class BigramNetwork:

    def __init__(self, seed=-1):
        if seed == -1:
            torch.randint(0, 9999999, (1, )).item()
        
        self.gen = torch.Generator().manual_seed(seed)
        self.data, self.itos, self.stoi = setup_data(parse_data_new())

        self._setup_network()

    def _setup_network(self) -> None:
        pass


def test_table_bigram():
    test = BigramWithTable()
    num_samples = 5
    for _ in range(num_samples):
        print(test.sample_name())


if __name__ == "__main__":
    test_table_bigram()

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

def setup_data(data) -> tuple:
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
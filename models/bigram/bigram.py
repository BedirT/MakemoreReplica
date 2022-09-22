import torch


class Bigram:

    def __init__(self, data):
        self.itos = {}
        self.stoi = {}
        self.data = data

        # populate maps
        all_data = set(''.join(self.data))
        self.itos[0] = '.' # beginning and ending token
        self.stoi['.'] = 0
        for i, v in enumerate(all_data):
            self.itos[i+1] = v
            self.stoi[v] = i+1
            
        self.generate_bigrams()
        


    def generate_bigrams(self) -> None: 
        # create a torch tensor of letter counters
        self.T = torch.zeros((len(self.stoi), len(self.stoi)), dtype=float)
        for sample in self.data:
            expanded_sample = ['.'] + list(sample) + ['.']
            for s1, s2 in zip(expanded_sample, expanded_sample[1:]):
                s1_id = self.stoi[s1]
                s2_id = self.stoi[s2]
                self.T[s1_id, s2_id] += 1
        
        # normalize each row
        for i, row in enumerate(self.T):
            tot = row.sum()
            row = row / tot
            self.T[i] = row

    def get_next(self) -> str:
        res = ''
        # pick the starting element
        row_idx = 0
        explored = 0
        while True:
            idx = torch.multinomial(self.T[row_idx], 1).item()
            if idx == 0 or explored > 20:
                break
            row_idx = idx
            res += self.itos[idx]
            explored += 1
        return res

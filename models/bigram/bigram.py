# Bigram implementations with count and neural network seperately.

import torch
import torch.nn.functional as F

from data_generation.data_tools import setup_data, parse_data_new


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
        self.probs = table / table.sum(dim=1, keepdim=True)

    def evaluate(self) -> float:
        # Evaluate the model using negative log likelihood
        log_likelihood = 0.0
        num_data_points = 0
        for name in self.data:
            tokenized_name = ['.'] + list(name) + ['.']
            for ch1, ch2 in zip(tokenized_name, tokenized_name[1:]):
                prob = self.probs[self.stoi[ch1], self.stoi[ch2]]
                log_likelihood += prob.log()
                num_data_points += 1
        return (-log_likelihood/num_data_points).item()

    def _get_next(self, idx:int) -> int:
        ''' Predict the next letter idx from the letter idx before '''
        return torch.multinomial(self.probs[idx], 
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

    def __init__(self, num_epochs=100, lr=50, seed=-1, verbose=True):
        if seed == -1:
            torch.randint(0, 9999999, (1, )).item()
        
        self.gen = torch.Generator().manual_seed(seed)
        self.num_epochs = num_epochs
        self.lr = lr
        self.verbose = verbose

        self.data, self.itos, self.stoi = setup_data(parse_data_new())

        self._setup_network()

    def _setup_network(self) -> None:
        # Setup dataset and train the network with generated samples
        Xs = []
        Ys = []
        for name in self.data:
            tokenized_name = ['.'] + list(name) + ['.']
            for ch1, ch2 in zip(tokenized_name, tokenized_name[1:]):
                Xs.append(self.stoi[ch1])
                Ys.append(self.stoi[ch2])
        Xs = torch.tensor(Xs)
        Ys = torch.tensor(Ys)
        
        # Training the network
        # initializing the weights
        self.Ws = torch.randn((len(self.stoi), len(self.stoi)), requires_grad=True) 

        num_unq_xs = Xs.nelement()
        xenc = F.one_hot(Xs, len(self.stoi)).float()
        for ep in range(self.num_epochs):
            # Forward pass
            logits = xenc @ self.Ws
            counts = torch.exp(logits)
            counts_sum = counts.sum(dim=1, keepdim=True)
            probs = counts / counts_sum
            
            # Loss
            loss = -probs[torch.arange(num_unq_xs), Ys].log().mean()
            loss += 0.01 * (self.Ws**2).mean() # regularization
            if self.verbose:
                print(f"Loss for epoch {ep}: {loss}")

            # Reset gradients
            self.Ws.grad = None
            loss.backward()

            # Update the gradients based on the loss
            self.Ws.data -= self.lr * self.Ws.grad

        # keeping the end loss
        self.loss = -probs[torch.arange(num_unq_xs), Ys].log().mean().item()

    def evaluate(self) -> float:
        return self.loss

    def _get_next(self, idx) -> int:
        xenc = F.one_hot(torch.tensor([idx]), num_classes=len(self.stoi)).float()
        logits = xenc @ self.Ws
        counts = logits.exp()
        counts_sum = counts.sum()
        probs = counts / counts_sum
        return torch.multinomial(probs, 1, generator=self.gen).item()

    def sample_name(self) -> str:
        idx = 0
        sampled_name = ""
        while True:
            idx = self._get_next(idx)
            if idx == 0:
                break
            sampled_name += self.itos[idx]
        return sampled_name


def test_table_bigram():
    test = BigramWithTable()
    num_samples = 5
    print("Generated names (using bigram table):")
    for _ in range(num_samples):
        print(' -> ' + test.sample_name())
    print("Loss:", test.evaluate())


def test_network_bigram():
    test = BigramNetwork(verbose=False)
    num_samples = 5
    print("Generated names (using bigram network):")
    for _ in range(num_samples):
        print(' -> ' + test.sample_name())
    print("Loss:", test.evaluate())


if __name__ == "__main__":
    test_table_bigram()
    test_network_bigram()
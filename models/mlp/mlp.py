import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random 

from data_generation.data_tools import setup_data, parse_data_new


class MakeMoreMLP:

    def __init__(self,
                 num_letter_seq=3,
                 num_dimensions=3,
                 num_hidden_neurons=100,
                 batch_size=32,
                 num_epochs=1000,
                 f_lr=lambda x: 0.1,
                 verbose=False,
                 print_every=1000):
        self.verbose = verbose
        self.num_letter_seq = num_letter_seq
        self.num_dimensions = num_dimensions
        self.num_hidden_neurons = num_hidden_neurons
        self.batch_size =  batch_size
        self.num_epochs = num_epochs
        self.f_lr = f_lr
        self.print_every = print_every

        self.data, self.itos, self.stoi = setup_data(parse_data_new())
        self.num_chars = len(self.stoi)

        # Just setting up the variables here to see them in constructor
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None

        self.set_data_split()
        self.train()

    def arrange_data(self, data):
        ''' Given data, aranges based on num of letters in a sequence,
        generates xs and ys. '''
        xs, ys = [], []
        for word in data:
            seq = [0] * self.num_letter_seq # initially all 0's (for '.')
            for c in word + '.':
                idx = self.stoi[c]
                xs.append(seq)
                ys.append(idx)
                # print(''.join([self.itos[i] for i in seq]), '--->', self.itos[idx])
                seq = seq[1:] + [idx] # sliding window
        return torch.tensor(xs), torch.tensor(ys)

    def set_data_split(self) -> None:
        ''' Sets up data split, 80-10-10 for train, val and test '''
        random.shuffle(self.data)
        limit_0, limit_1 = int(len(self.data) * .8), int(len(self.data) * .9)
        self.x_train, self.y_train = self.arrange_data(self.data[:limit_0])
        self.x_val, self.y_val = self.arrange_data(self.data[limit_0:limit_1])
        self.x_test, self.y_test = self.arrange_data(self.data[:limit_1])

    def setup_network(self) -> None:
        self.C = torch.randn((self.num_chars, self.num_dimensions))
        self.W1 = torch.randn((self.num_dimensions * self.num_letter_seq, self.num_hidden_neurons))
        self.b1 = torch.randn(self.num_hidden_neurons)
        self.W2 = torch.randn((self.num_hidden_neurons, self.num_chars)) # output layer
        self.b2 = torch.randn((self.num_chars))
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

    def get_num_parameters(self) -> int:
        return sum(p.nelement() for p in self.parameters)

    def require_grads(self) -> None:
        for p in self.parameters:
            p.requires_grad = True

    def reset_grads(self) -> None:
        for p in self.parameters:
            p.grad = None

    def train(self) -> None:
        ''' Trains the model '''
        self.setup_network()
        self.require_grads()

        for e in range(self.num_epochs):
            # forward pass
            # sample a mini batch
            idxs = torch.randint(0, self.x_train.shape[0], (self.batch_size,))
            emb = self.C[self.x_train[idxs]]
            h = torch.tanh(emb.view(-1, self.num_dimensions * self.num_letter_seq) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2

            # backward pass
            loss = F.cross_entropy(logits, self.y_train[idxs]) # nll
            self.reset_grads()
            loss.backward()
            if self.verbose and e % self.print_every == 0:
                print(f"Ep {e}: Training loss --> {loss}")

            # update
            lr = self.f_lr(e)
            for p in self.parameters:
                p.data += -lr * p.grad

    def get_loss(self, data_split: str):
        splits = {
            'train': (self.x_train, self.y_train),
            'val': (self.x_val, self.y_val),
            'test': (self.x_test, self.y_test)
        }[data_split]
        emb = self.C[splits[0]]
        h = torch.tanh(emb.view(-1, self.num_dimensions * self.num_letter_seq) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, splits[1])
        return loss.item()

    def get_loss(self) -> dict:
        return {'train_loss': self.get_loss('train'),
                'val_loss': self.get_loss('val')}
        
    def sample_words(self, num_words) -> str:
        # Sampling from the model
        outs = []
        for _ in range(num_words):
            out = ''
            context = [0] * self.num_letter_seq
            while len(out) < 20: # safe guard
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                idx = torch.multinomial(probs, 1).item()
                context = context[1:] + [idx]
                if idx == 0:
                    break
                out += self.itos[idx]
            outs.append(out)
        return outs

        
if __name__ == "__main__":
    num_epochs = 200000
    mlp = MakeMoreMLP(
        num_letter_seq=5,
        num_dimensions=10,
        num_hidden_neurons=300,
        batch_size=128,
        num_epochs=num_epochs,
        f_lr=lambda x: 0.1 if x < num_epochs/2 else 0.01,
        verbose=False,
        print_every=10000)
    print(f"Final loss: {mlp.get_loss()}")
    print(f"Number of parameters: {mlp.get_num_parameters()}")
    print(f"Sample words: {mlp.sample_words(10)}")

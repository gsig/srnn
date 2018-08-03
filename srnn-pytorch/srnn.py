import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tensorchoice(n, scores):
    p = scores.detach().numpy()
    p = p / p.sum()
    return np.random.choice(n, p=p)


def tensormax(n, scores):
    p = scores.detach().numpy()
    return np.argmax(p)


def pick_probabilities(k, numberspicked, n, current):
    # probabilities of picking each of the future elements
    # if we are picking k elements sequentially without replacement
    prob = torch.zeros(n)
    no = 1.
    remaining_picks = k - numberspicked
    for i in range(current, n - remaining_picks + 1):
        s = remaining_picks / float(n - i)
        prob[i] = s * no
        no = no * (1 - s)
    return prob


class SRNN(nn.Module):
    # This should follow similar semantics as nn.LSTM
    def __init__(self, input_dim, hidden_dim, subset=10,
                 input_fun=lambda x: x,
                 output_fun=lambda x: x,
                 similarity=lambda x, y: (x * y).sum(2).sum(1),
                 choice=tensormax):
        super(SRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.input_fun = input_fun
        self.output_fun = output_fun
        self.similarity = similarity
        self.choice = choice
        self.hidden = self.init_hidden()
        self.output = self.init_output()
        self.subset = subset

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def scores(self, embeds, numberspicked, current):
        n = len(embeds)
        scores = self.similarity(self.output, embeds)
        scores = F.softmax(scores, dim=0)
        scores = scores * pick_probabilities(self.subset, numberspicked, n, current)
        scores = scores / scores.sum()
        return scores

    def init_output(self):
        return self.output_fun(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input):
        n = len(input)
        embeds = self.input_fun(input)

        # pick first node
        scores = self.scores(embeds, 0, 0)
        choice = self.choice(n, scores)
        picks = [choice]
        loss = -torch.log(scores[choice]) / n
        outputs = []

        for i, e in enumerate(embeds):
            outputs.append(self.output)
            if picks[-1] > i:
                # skip elements until next node
                continue
            lstm_out, self.hidden = self.lstm(e.view(1, 1, -1), self.hidden)
            self.output = self.output_fun(lstm_out.view(1, -1))

            if len(picks) < self.subset:
                # pick next node
                scores = self.scores(embeds, len(picks), i + 1)
                choice = self.choice(n, scores)
                picks.append(choice)
                loss -= torch.log(scores[choice]) / (n - i)
        return loss, outputs, picks

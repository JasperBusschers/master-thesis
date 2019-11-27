import torch
import torch.nn as nn
import torch.nn.functional as F




class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(state_dim +action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.loss_fn = nn.BCELoss()

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

    def update(self,batch_expert,batch_agent):
        self.optim_discriminator.zero_grad()
        batch_size = len(batch_agent)
        # label tensors
        exp_label = torch.full((batch_size, 1), 1, )
        policy_label = torch.full((batch_size, 1), 0, )

        # with expert transitions
        prob_exp = self.discriminator(batch_expert)
        loss = self.loss_fn(prob_exp, exp_label)

        # with policy transitions
        prob_policy = self.discriminator(batch_agent)
        loss += self.loss_fn(prob_policy, policy_label)

        # take gradient step
        loss.backward()
        self.optim_discriminator.step()


class discriminator_module():
    def init(self, state_size,action_size,args):
        self.discriminators = [Discriminator(state_size,action_size) for _ in range(args.amount_of_disc)]


    def update(self, memory, batch_size, policy):
        batch_expert = memory.sample(batch_size)
        batch_expert = torch.FloatTensor(batch_expert)


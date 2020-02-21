import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(state_dim +action_dim, 24)
        self.l2 = nn.Linear(24, 88)
        self.l3 = nn.Linear(88, 1)
        self.optim_discriminator = torch.optim.Adam(self.parameters(), lr=args.disc_lr)
        self.loss_fn = nn.BCELoss()
        self.args = args

    def forward(self, sample):
        x = torch.tanh(self.l1(sample))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

    def update(self,batch_expert,batch_agent):
        self.optim_discriminator.zero_grad()
        batch_size = self.args.batch_size
        # label tensors
        exp_label = torch.full((batch_size, 1), 1, )
        policy_label = torch.full((batch_size, 1), 0, )

        # with expert transitions
        prob_exp = self.forward(batch_expert)
        loss = self.loss_fn(prob_exp, exp_label)

        # with policy transitions
        prob_policy = self.forward(batch_agent)
        loss += self.loss_fn(prob_policy, policy_label)

        # take gradient step
        loss.backward()
        self.optim_discriminator.step()
        return loss.item()


class discriminator_module():
    def __init__(self, args):
        self.discriminators = [Discriminator(args.number_of_steps ,args.number_of_steps , args) for _ in range(args.amount_of_disc)]
        self.args = args
        self.discriminator_losses = [[] for _ in range(args.amount_of_disc)]


    def update(self, memory , disc_index, memory_index):
        batch_size = self.args.batch_size
        batch_expert = memory.sample_dom_buffer(batch_size,memory_index,also_agent=True)
        agent_batch = memory.sample_experiences(batch_size)
        batch_expert = torch.FloatTensor(batch_expert)
        agent_batch = torch.FloatTensor(agent_batch)
        loss = self.discriminators[disc_index].update(batch_expert,agent_batch)
        self.discriminator_losses[disc_index].append(loss)

    def get_losses(self):
        return self.discriminator_losses

    def get_reward(self,sample,method,memory):
        total = 0
        sample = torch.FloatTensor(sample)
        if method == 'sum':
            batch_expert = memory.sample_dom_buffer(1, 1, also_agent=True)
            batch_expert = torch.FloatTensor(batch_expert).view(-1)
            for i, disc in enumerate(self.discriminators):
                total += disc(sample).item() - disc(batch_expert).item()
            #total = total-2
        if method == 'logsum':
            for disc in self.discriminators:
                total += torch.log(disc(sample)+0.3).item()
        if method == 'logsumdiff':
            for disc in self.discriminators:
                total += (torch.log(disc(sample)) - torch.log(1-disc(sample))).item()
        return total

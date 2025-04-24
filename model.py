import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator
from torch.nn import Module
import torch.nn.functional as F


class DMIGNN(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(DMIGNN, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.interests = opt.interests
        self.length = opt.length
        self.beta = opt.beta

        self.adj_all = trans_to_cuda(torch.Tensor(adj_all), self.device).long()
        self.num = trans_to_cuda(torch.Tensor(num), self.device).float()

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.interests))
        self.interest_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.Tanh(),
                nn.Linear(self.dim, self.dim)
            ) for _ in range(self.interests)
        ])
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        # 使用独立提取器提取不同兴趣
        interest_vectors = []
        for i in range(self.interests):
            # 对每个位置独立应用提取器
            # nh形状为[batch_size, seq_len, dim]
            interest_i = self.interest_extractors[i](nh)  # [batch_size, seq_len, dim]
            interest_vectors.append(interest_i)

        # 堆叠所有兴趣向量
        # 形状变为[batch_size, seq_len, dim, interests]
        nh = torch.stack(interest_vectors, dim=3)

        w2 = self.w_2.unsqueeze(0)
        w2 = w2.repeat(nh.shape[0], nh.shape[1], 1, 1)

        beta = torch.sum(nh * w2, dim=2)

        mask = mask.expand(-1, -1, self.interests)

        beta = beta * mask

        sumask = torch.sum(mask, 1, )
        sumask.to(torch.int)

        # 创建一个包含维度索引的列表
        dimensions = list(range(self.interests))

        # 对于每个维度，进行归一化
        normalized_beta = torch.empty_like(beta).to(self.device)
        for i in dimensions:
            normalized_beta[:, :, i] = F.normalize(beta[:, :, i], p=2, dim=1)

        lens = sumask[:, 0] - self.length

        sim_loss = torch.zeros(nh.shape[0], dtype=torch.float32).to(self.device)
        for i in dimensions[:-1]:
            for j in dimensions[i + 1:]:
                temp_sim = torch.sum(normalized_beta[:, :, i] * normalized_beta[:, :, j], dim=1)
                temp_sim = torch.abs(temp_sim)
                sim_loss += temp_sim

        sim_loss = sim_loss * 2 / (self.interests * (self.interests - 1))
        loss1 = sim_loss * lens
        loss1 = torch.sigmoid(loss1)
        loss1 = torch.sum(loss1, dim=-1)

        selects = []
        for i in dimensions:
            selects.append(torch.sum(beta[:, :, i].unsqueeze(-1) * hidden, 1))

        select = torch.stack(selects, dim=0)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        # added
        b = F.normalize(b, p=2.0, dim=-1)
        scores = torch.matmul(select, b.transpose(1, 0))
        sum_scores = torch.sum(scores, dim=0)
        max_scores, max_indices = torch.max(scores, dim=0)

        return max_scores, loss1 * self.beta, sum_scores

    def forward(self, inputs, adj, mask_item, item):
        h = self.embedding(inputs)
        # added
        h = F.normalize(h, p=2.0, dim=-1)
        # local
        h_local = self.local_agg(h, adj, mask_item)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        output = h_local

        return output


def trans_to_cuda(variable, device):
    if torch.cuda.is_available():
        return variable.cuda(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, device):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs, device).long()
    items = trans_to_cuda(items, device).long()
    adj = trans_to_cuda(adj, device).float()
    mask = trans_to_cuda(mask, device).long()
    inputs = trans_to_cuda(inputs, device).long()
    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, device):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, (scores, loss1, _) = forward(model, data, device)
        targets = trans_to_cuda(targets, device).long()
        loss = model.loss_function(scores, targets - 1) + loss1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    cov = set()
    for data in test_loader:
        targets, (scores, _, sum_scores) = forward(model, data, device)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            cov.update(score)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    result.append(len(cov))

    return result

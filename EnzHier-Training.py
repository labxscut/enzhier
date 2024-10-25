import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
import torch.nn.functional as F

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args

def random_positive2(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    
    # 如果只有一个酶序列，直接返回该酶序列的ID
    if len(ec_id[pos_ec]) == 1:
        return random.choice(ec_id[pos_ec])
    
    # 如果有多个酶序列，随机选择一个不等于给定ID的酶序列的ID
    while True:
        pos = random.choice(ec_id[pos_ec])
        if pos != id:
            return pos

class Triplet_dataset_with_mine_EC2(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg):

        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        
        # random.seed(1234)
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        
        pos = random_positive2(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load('./data/esm_data/' + anchor + '.pt')
        p = torch.load('./data/esm_data/' + pos + '.pt')
        n = torch.load('./data/esm_data/' + neg + '.pt')
        return format_esm(a), format_esm(p), format_esm(n), self.id_ec[anchor], self.id_ec[neg], self.id_ec[pos]
    
def collate_fn(batch):
    samples1, samples2, samples3, label1, label2, label3 = zip(*batch)
    return (samples1, samples2, samples3), label1, label2, label3

def get_dataloader2(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
        'collate_fn': collate_fn  # 设置collate_fn参数
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC2(id_ec, ec_id, negative)    
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def calculate_ec_similarity(ec1_list, ec2_list):
    ec1_levels_list = [ec.split('.') for ec in ec1_list]
    ec2_levels_list = [ec.split('.') for ec in ec2_list]
    
    # 定义不同层级的权重
    weights = [4, 3, 2, 1]  # 假设第一位权重为 4，依次递减
    
    # 计算相似度
    similarities = []
    for ec1_levels, ec2_levels in zip(ec1_levels_list, ec2_levels_list):
        similarity = 0
        for i in range(min(len(ec1_levels), len(ec2_levels))):
            if ec1_levels[i] == ec2_levels[i]:
                similarity += weights[i]
            else:
                break  # 如果出现不同等级，停止计算相似度
        # 归一化相似度
        max_levels = max(len(ec1_levels), len(ec2_levels))
        similarity /= sum(weights[:max_levels])
        similarities.append(similarity)
        
    return torch.tensor(similarities)

class TripletMarginLossWithEC(torch.nn.Module):
    def __init__(self, margin_max, margin_base):
        super(TripletMarginLossWithEC, self).__init__()
        self.margin_max = margin_max
        self.margin_base = margin_base

    def forward(self, anchor_embed, positive_embed, negative_embed, ec_sim_ap, ec_sim_an):
        # 计算 EC 号的损失
        ec_margin = self.margin_base + (self.margin_max - self.margin_base) * (ec_sim_ap - ec_sim_an)
        
        # 计算序列嵌入的损失
        seq_loss = F.relu(torch.norm(anchor_embed - positive_embed, p=2, dim=1) ** 2 - torch.norm(anchor_embed - negative_embed, p=2, dim=1) ** 2 + ec_margin)
        
        loss = seq_loss
        return torch.mean(loss)

def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    loss_values = []  # 用于保存每个batch的损失值

    for batch, (samples, ec_a, ec_an, ec_ap) in enumerate(train_loader):
        optimizer.zero_grad()

        anchor, positive, negative = samples
        ec_a = [item[0] for item in ec_a]                                 #多EC号的情况下，只选择首个EC
        ec_ap = [item[1] if len(item) > 1 else item[0] for item in ec_a]     #多EC号的情况下，只选择第2个EC
        ec_an = [item[0] for item in ec_an]         #多EC号的情况下，只选择首个EC
        
        anchor = torch.stack(anchor, dim=0).to(device=device, dtype=dtype)
        positive = torch.stack(positive, dim=0).to(device=device, dtype=dtype)
        negative = torch.stack(negative, dim=0).to(device=device, dtype=dtype)
        
        ec_sim_ap = calculate_ec_similarity(ec_ap, ec_ap)  # 相同的 EC 号，相似度为 1
        ec_sim_an = calculate_ec_similarity(ec_an, ec_ap)
        ec_sim_ap = ec_sim_ap.to(device=device, dtype=dtype)
        ec_sim_an = ec_sim_an.to(device=device, dtype=dtype)
        
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
      
        loss = criterion(anchor_out, positive_out, negative_out, 
                         ec_sim_ap.unsqueeze(0), ec_sim_an.unsqueeze(0))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        loss_values.append(loss.item())  # 将损失值保存到列表中
        
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1), loss_values

#================================================================================
#================================================================================

def main():
    seed_everything()
    ensure_dirs('./data/model')
    
    args = parse()
    
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
 
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    criterion = TripletMarginLossWithEC(margin_max=1.0, margin_base=0.8)
    best_loss = float('inf')
    train_loader = get_dataloader2(dist_map, id_ec, ec_id, args)
  
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training
    for epoch in range(1, epochs + 1):
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss, loss_values = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # remove tmp save weights
    # os.remove('./data/model/' + model_name + '.pth')
    # os.remove('./data/model/' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), './data/model/' + model_name + '.pth')


if __name__ == '__main__':
    main()

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model import NeuMF, GMF, MLP
from utils import date, predict_mse, NCFDataset


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = predict_mse(model, train_dataloader, config.device)
    valid_mse = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss = 100
    for epoch in range(config.train_epochs):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_id, item_id, ratings = [i.to(config.device) for i in batch]
            predict = model(user_id, item_id)
            loss = F.mse_loss(predict, ratings, reduction='mean')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  # 停止训练状态
        valid_mse = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, next(model.parameters()).device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


def main():
    config = Config()
    print(config)

    df = pd.read_csv(config.dataset_file, usecols=[0, 1, 2])
    df.columns = ['userID', 'itemID', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()
    user_count = df['userID'].value_counts().count()  # 用户数量
    item_count = df['itemID'].value_counts().count()  # item数量
    print(f"{date()}## Dataset contains {user_count} users and {item_count} items.")

    train_data, valid_data = train_test_split(df, test_size=1 - 0.8, random_state=3)  # split dataset including random
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=4)
    train_dataset = NCFDataset(train_data)
    valid_dataset = NCFDataset(valid_data)
    test_dataset = NCFDataset(test_data)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    os.makedirs('./weights', exist_ok=True)  # 文件夹不存在则创建
    print(f'{date()}############ 预训练GMF ###########################')
    model_GMF = GMF(user_count, item_count, config.mf_dim).to(config.device)
    train(train_dlr, valid_dlr, model_GMF, config, 'weights/GMF.pt')
    test(test_dlr, torch.load('weights/GMF.pt'))

    print(f'{date()}############ 预训练MLP ###########################')
    model_MLP = MLP(user_count, item_count, config.mlp_layers).to(config.device)
    train(train_dlr, valid_dlr, model_MLP, config, 'weights/MLP.pt')
    test(test_dlr, torch.load('weights/MLP.pt'))

    print(f'{date()}############ 训练NeuMF ###########################')
    model_NeuMF = NeuMF(user_count, item_count, config.mf_dim, config.mlp_layers, use_pretrain=True).to(config.device)
    train(train_dlr, valid_dlr, model_NeuMF, config, 'weights/NeuMF.pt')
    test(test_dlr, torch.load('weights/NeuMF.pt'))


if __name__ == '__main__':
    main()

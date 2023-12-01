from sklearn.model_selection import KFold
from ast import parse
import os
import importlib
import argparse
import numpy as np
import pandas as pd
import data_utils
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal


from itertools import cycle
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

from data_utils import  get_data
from utils import mse_loss_with_nan, ramp_up, augment, likelihood_with_mask, kl_div_with_mask
import time

parser = argparse.ArgumentParser()
parser.add_argument('--window', type=int, help='window size', required=True)
parser.add_argument('--patch', type=int, help='patch size', required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--percent', type=float, default=1.0, help='Percentage of samples with maximum non-missing rate')
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--stage', type=str, default='train')
parser.add_argument('--model_path', type=str, help='When stage is test, load trained model from model_path')
args = parser.parse_args()
# python pipeline.py --window 1 --patch 18 --model cnn --percent 1 --stage train
# python pipeline.py --window 12 --patch 18 --model cnn --percent 1 --stage train
pro_name = ['sss', 'ssh','slp', 'wind', 'mld', 'sst', 'ice', 'co2', 'chl']
class Pipeline():
    def __init__(self):
        target = 'IPSL'
        config = {
            'batch_size': 16,
            'epochs': 100,
            'lr': 2e-4,
            'weight_decay': 1e-3,
            'seed': 0,
            'early_stop_round': 30,
            'save_model': False,
            'use_cuda': True,
            'dump_dir': './checkpoint/',
            'result_dir': './result/',
            'preprocessed_data_dir': 'dataset/preprocessed_data/',
            'split_data_dir': 'dataset',
            'columns': ['year', 'month', 'latitude', 'longitude', target,'sss', 'ssh','slp', 'wind', 'mld', 'sst', 'ice', 'co2', 'chl'],
            'norm_columns': [ target, 'sss', 'ssh','slp', 'wind', 'mld', 'sst', 'ice', 'co2', 'chl'],
            'input_dim': 13,
            'target' : target,
            'exp_id': 1
        }
        self.config = config
        self.config.update(args.__dict__)

        torch.manual_seed(config['seed'])
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        self.normalizer = None
        self.models = []
    
        data = get_data(mode='new', config=self.config)
        
        # build train and valid dataloader
        train_data = data[(2018 <= data.year) & (data.year <= 2019)]
        valid_data = data[(2020 <= data.year) & (data.year <= 2021)]

        valid_data = data_utils.add_prefix(
            valid_data, train_data, config['window'])

        train_loader = self.build_dataloader(
            mode='train', config=self.config, data=train_data, shuffle=True)
        valid_loader = self.build_dataloader(
            mode='valid', config=self.config, data=valid_data, shuffle=False)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # build test dataloader
        data = get_data(mode='new', config=self.config)
        data = data[(1959 <= data.year) & (data.year <= 2021)]
        test_loader = self.build_dataloader(
            mode='valid', config=self.config, data=data, shuffle=False)
        self.test_loader = test_loader

    def normalize(self, data):
        if self.normalizer is None:
            self.normalizer = {}
            for col in self.config['norm_columns']:
                self.normalizer[col] = RobustScaler()
                self.normalizer[col].fit(data[col].values.reshape(-1, 1))

        for col in self.config['norm_columns']:
            data[col] = self.normalizer[col].transform(
                data[col].values.reshape(-1, 1)
            ).reshape(-1, )

        return data

    def build_dataloader(self, mode, config, data, shuffle=False):
        print('normalize data...')
        data = self.normalize(data.copy())
        data[pro_name] = data[pro_name].fillna(0.0)

        print('build dataloader')
        dataloader = data_utils.dataloader(
            data[self.config['columns']], self.config, n_jobs=0, shuffle=shuffle, mode=mode
        )
        print('build dataloader finished')
        return dataloader

    def run_epoch(self, model, data_loader, optimizer, epoch):
        model.train()

        avg_sup_loss = 0.0
        avg_unsup_label_loss = 0.0

        with tqdm(data_loader, mininterval=3, disable=args.disable_tqdm) as bar:
            for batch_id, batch in enumerate(bar):
                data, target = batch

                data = data.to(self.device)
                target = target.to(self.device)
                
                # weak aug
                data_w = augment(data, method='mask', intensity=0.1)
                # strong aug
                data_s = augment(data, method='mask', intensity=0.3)

                mask = 1 - torch.isnan(target).float()

                # supervised part
                out_w = model(data_w)
                sup_loss = mse_loss_with_nan(out_w, target, mask=mask)

                # unsupervised part
                out_s = model(data_s)
                unsup_loss = mse_loss_with_nan(out_s, out_w, mask=1-mask)

                loss = sup_loss + unsup_loss * 0.2

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                avg_sup_loss += sup_loss.item()
                avg_unsup_label_loss += unsup_loss.item()

                bar.set_postfix(
                    avg_label_loss=avg_sup_loss/(batch_id + 1),
                    avg_pseudo_label_loss=avg_unsup_label_loss/(batch_id + 1),
                )

    def eval_on(self, model, dataloader):
        model.eval()

        index = []
        predictions = []
        targets = []

        for batch_id, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                pred = model(data)
            index.append(data[..., -1, :4].data.cpu().numpy())
            predictions.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        index = np.concatenate(index, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        index = index.reshape(-1, index.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, target.shape[-1])

        predictions = self.normalizer[self.config["target"]].inverse_transform(predictions).reshape(-1, )
        targets = self.normalizer[self.config["target"]].inverse_transform(targets).reshape(-1, )
        result = pd.DataFrame({
            'year': index[:, 0],
            'month': index[:, 1],
            'lat': index[:, 2],
            'lon': index[:, 3],
            'predictions': predictions,
            'targets': targets
        })
        result = result.groupby(['year', 'month', 'lat', 'lon']).mean()

        eval_result = result.dropna()

        loss = np.mean(
            (eval_result['predictions'] - eval_result['targets']) ** 2
        )

        return loss

    def run(self):
        # initialize student model
        model_class = importlib.import_module(
            f'model.{self.config["model"]}'
        )
        model = model_class.Net(self.config).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        self.best_val = np.inf
        self.early_stop_cnt = 0
        self.best_model = None

        patience = self.config['early_stop_round']

        print('start training...')

        for epoch in range(self.config['epochs']):
            self.run_epoch(model, self.train_loader, optimizer, epoch)
            val_metric = self.eval_on(model, self.valid_loader)

            print(f'Epoch {epoch}, validation metric {val_metric}')

            if val_metric < self.best_val:
                self.best_val = val_metric
                self.best_model = copy.deepcopy(model)
                patience = self.config['early_stop_round']
            else:
                patience -= 1

            if patience == 0:
                break
            
        self.models.append(model)
        if self.config['dump_dir']:
            os.makedirs(self.config['dump_dir'], exist_ok=True)
            torch.save(self.best_model.state_dict(), os.path.join(
                self.config['dump_dir'], f'{self.config["exp_id"]}_{time.strftime("%Y-%m-%d_%H:%M", time.localtime())}_{self.best_val:.2f}.pth'))
    
    def test(self):
        print('test start...')
        # load model
        if hasattr(self, 'best_model') and self.best_model:
            model = self.best_model
        else:
            assert (hasattr(self.config, 'model_path') and isinstance(
                self.config['model_path'], str)), 'model_path should be provided'
            model_class = importlib.import_module(
                f'model.{self.config["model"]}'
            )
            model = model_class.Net(self.config).to(self.device)
            model.load_state_dict(torch.load(self.config['model_path']))

        model.eval()

        test_data = get_data(mode='test', config=self.config)
        test_loader = self.build_dataloader(
            mode='test', config=self.config, data=test_data, shuffle=False)

        index = []
        predictions = []
        targets = []

        for batch_id, (data, target) in enumerate(test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                pred = model(data)

            # year, month, lat, lon
            index.append(data[..., -1, :4].data.cpu().numpy())
            predictions.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        index = np.concatenate(index, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        index = index.reshape(-1, index.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, target.shape[-1])

        predictions = self.normalizer[self.config["target"]].inverse_transform(predictions).reshape(-1, )
        targets = self.normalizer[self.config["target"]].inverse_transform(targets).reshape(-1, )

        result = pd.DataFrame({
            'year': index[:, 0],
            'month': index[:, 1],
            'lat': index[:, 2],
            'lon': index[:, 3],
            'predictions': predictions,
            'targets': targets
        })
        result = result.groupby(['year', 'month', 'lat', 'lon']).mean()
        result.to_csv('./result/result.csv')
        print('test finished...')

if __name__ == '__main__':
    mt = Pipeline()  # supervised/semi_supervised
    if mt.config['stage'] == 'train':
        mt.run()
    elif mt.config['stage'] == 'test':
        mt.test()
    else:
        print('invalid stage')

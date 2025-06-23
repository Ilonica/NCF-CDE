import torch
import os
import glob
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from checkpoint_manager import (
    save_checkpoint,
    resume_checkpoint,
    find_last_checkpoint,
    convert_checkpoint,
    use_optimizer
)
from metrics import MetronAtK

# Engine реализует весь цикл обучения модели NCF-CDE
class Engine(object):
    def __init__(self, config):
        self.config = config
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCELoss()

    # Обработка одного батча данных
    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Пожалуйста, укажите точную модель!'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    # Обучение модели одну эпоху
    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Укажите точную модель!'
        self.model.train()
        total_loss = 0
        train_loader_iter = tqdm(train_loader, desc=f"Эпоха обучения {epoch_id}")
        for batch_id, batch in enumerate(train_loader_iter):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            train_loader_iter.set_postfix(loss=f"{loss:.4f}")
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        return total_loss

    # Оценка модели на тестовых данных
    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Укажите точную модель!'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]

            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

            test_scores = []
            negative_scores = []

            # Обработка батчей для тестовых данных
            bs = self.config['batch_size'] if self.config['use_bachify_eval'] else len(test_users)
            test_batches = [(start, min(start + bs, len(test_users)))
                            for start in range(0, len(test_users), bs)]

            for start_idx, end_idx in tqdm(test_batches, desc=f"Evaluating - Test"):
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(self.model(batch_test_users, batch_test_items))

            # Обработка батчей для негативных данных
            neg_batches = [(start, min(start + bs, len(negative_users)))
                           for start in range(0, len(negative_users), bs)]

            for start_idx, end_idx in tqdm(neg_batches, desc=f"Evaluating - Negative"):
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(self.model(batch_negative_users, batch_negative_items))

            # Объединение результатов
            test_scores = torch.cat(test_scores, dim=0)
            negative_scores = torch.cat(negative_scores, dim=0)

            # Перенос данных на CPU, если используется CUDA
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()

            # Установка subjects для MetronAtK
            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                negative_users.data.view(-1).tolist(),
                negative_items.data.view(-1).tolist(),
                negative_scores.data.view(-1).tolist()
            ]

        # Расчет метрик
        hit_ratio = self._metron.cal_hit_ratio()
        ndcg = self._metron.cal_ndcg()

        # Логирование в TensorBoard
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)

        return hit_ratio, ndcg

    # Сохранение чекпоинта (состояния обучения)
    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Укажите точную модель!'

        checkpoint = {
            'epoch': epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'hr': hit_ratio,
            'ndcg': ndcg,
            'config': self.config
        }

        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        torch.save(checkpoint, model_dir)
        return model_dir

    # Загрузка последнего чекпоинта
    def load_last_checkpoint(self, alias):
        last_checkpoint = find_last_checkpoint(alias)
        if last_checkpoint:
            return resume_checkpoint(
                self.model,
                last_checkpoint,
                self.config['device_id'],
                self.opt
            )
        return None

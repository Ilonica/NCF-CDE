import torch
from engine import Engine
from checkpoint_manager import use_cuda
from torch import nn

class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        # Эмбеддинги пользователей и товаров
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Выходной слой
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # Инициализация весов нормальным распределением
        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    print(sm)
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    # Прямой проход модели
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding) # Поэлементное произведение
        logits = self.affine_output(element_product) # Прогнозирование рейтинга
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class GMFEngine(Engine):
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)
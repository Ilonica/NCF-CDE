import torch
from torch import nn
from torchdiffeq import odeint
from gmf import GMF
from engine import Engine
from checkpoint_manager import use_cuda, resume_checkpoint
from torchdiffeq import odeint

class CDECF(nn.Module):
    def __init__(self, config, user_item_interactions):
        super(CDECF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.device = torch.device(f"cuda:{config['device_id']}" if config['use_cuda'] else "cpu")

        # Строим матрицу смежности
        self.adj_matrix = self.build_joint_adjacency(user_item_interactions).to(self.device)
        self.adj_matrix = self.normalize_adj(self.adj_matrix)

        # Эмбеддинги
        self.user_emb = nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim).to(self.device)

        # Neural ODE
        self.ode_func = ODEFunc(self.latent_dim, self.adj_matrix, self.num_users).to(self.device)
        self.time_steps = torch.linspace(0, config['ode_time'], steps=config['ode_steps'], device=self.device)

        self.init_weights()

    # Строим объединенную разреженную матрицу смежности
    def build_joint_adjacency(self, user_item_sparse):
        num_nodes = self.num_users + self.num_items
        rows, cols = user_item_sparse.indices()
        values = user_item_sparse.values()

        # Смещаение индексов товаров
        cols = cols + self.num_users

        # Транспонированные индексы
        trans_rows = cols
        trans_cols = rows

        # Объединение индексов
        all_rows = torch.cat([rows, trans_rows])
        all_cols = torch.cat([cols, trans_cols])
        all_values = torch.cat([values, values])

        # Создаем объединенную sparse матрицу
        indices = torch.stack([all_rows, all_cols])
        return torch.sparse_coo_tensor(
            indices,
            all_values,
            size=(num_nodes, num_nodes)
        ).coalesce()

    # Нормировка разреженной матрицы смежности
    def normalize_adj(self, adj):
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5).view(-1)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()

        # Умножение разреженных матриц
        tmp = torch.sparse.mm(adj, d_mat_inv_sqrt)
        return torch.sparse.mm(d_mat_inv_sqrt, tmp)

    # Инициализация весов эмбеддингов нормальным распределением
    def init_weights(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

    # Прямой проход модели
    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        E_u = self.user_emb(users)
        E_i = self.item_emb(items)

        # Решение Neural ODE
        ode_input = torch.cat([E_u, E_i], dim=1)
        ode_output = odeint(
            self.ode_func,
            ode_input,
            self.time_steps,
            method=self.config['ode_solver']
        )[-1]

        # Разделение обновленных эмбеддингов
        E_u_final, E_i_final = ode_output.chunk(2, dim=1)

        # Вычисление предсказаний
        predictions = torch.sum(E_u_final * E_i_final, dim=1)
        predictions = torch.sigmoid(predictions)
        # return predictions

        return E_u_final * E_i_final

    # Вычисление Bayesian Personalized Ranking loss.
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        # Добавление L2-регуляризации
        if self.config['weight_decay'] > 0:
            l2_reg = (self.user_emb.weight.norm() + self.item_emb.weight.norm()) * self.config['weight_decay']
            loss += l2_reg
        return loss

    # Загрузка предобученных весов для GMF части
    def load_pretrain_weights(self):
        if self.config['pretrain_gmf']:
            print(f"Loading pretrained weights from {self.config['pretrain_gmf']}")
            pretrained_model = torch.load(self.config['pretrain_gmf'])
            self.user_emb.weight.data.copy_(pretrained_model['user_emb.weight'])
            self.item_emb.weight.data.copy_(pretrained_model['item_emb.weight'])


class CDECFEngine(Engine):
    def __init__(self, config):
        # Загрузка взаимодействия пользователь-товар
        user_item_interactions = self.load_interactions(config)

        # Инициализация модели
        self.model = CDECF(config, user_item_interactions)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(CDECFEngine, self).__init__(config)

        if config['pretrain']:
            self.model.load_pretrain_weights()

    # Генеририя случайных взаимодействий для инициализации
    def load_interactions(self, config):
        indices = torch.randint(0, config['num_items'], (config['num_users'] * 10,))
        rows = torch.randint(0, config['num_users'], (config['num_users'] * 10,))
        values = torch.ones(config['num_users'] * 10)

        return torch.sparse_coo_tensor(
            torch.stack([rows, indices]),
            values,
            size=(config['num_users'], config['num_items'])
        ).coalesce()

# Функция для Neural ODE, моделирующая динамику эмбеддингов на графе
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, adj_matrix, num_users):
        super().__init__()
        self.latent_dim = latent_dim
        self.adj_matrix = adj_matrix
        self.num_users = num_users
        self.device = adj_matrix.device

        # Сеть для генерации весовых коэффициентов
        self.weight_generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        ).to(self.device)

   # Вычислние производных эмбеддингов в момент времени t
    def forward(self, t, x):
        batch_size = x.size(0)
        E_u, E_i = x[:, :self.latent_dim], x[:, self.latent_dim:]

        # Формируем эмбеддинги полного графа (пользователи + айтемы)
        full_embeddings = torch.zeros(self.num_users + (self.adj_matrix.size(0) - self.num_users),
                                       self.latent_dim, device=self.device)

        # Заполняем эмбеддинги только юзеров и айтемов из батча
        full_embeddings_batch_indices = torch.cat([
            torch.arange(0, E_u.size(0), device=self.device),
            torch.arange(self.num_users, self.num_users + E_i.size(0), device=self.device)
        ])
        full_embeddings[full_embeddings_batch_indices] = torch.cat([E_u, E_i], dim=0)

        # Разреженное матричное произведение
        graph_effect = torch.sparse.mm(self.adj_matrix, full_embeddings)


        graph_effect_u = graph_effect[:E_u.size(0)]
        graph_effect_i = graph_effect[self.num_users:self.num_users + E_i.size(0)]

        # Удаляем self-loop влияние
        effect_u = graph_effect_u - E_u
        effect_i = graph_effect_i - E_i

        # Генерация весов
        weights = self.weight_generator(x)

        # Применение весов
        dE_u = weights * effect_u
        dE_i = weights * effect_i

        return torch.cat([dE_u, dE_i], dim=1)




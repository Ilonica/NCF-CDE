import torch
from gmf import GMF
from mlp import MLP
from ode import CDECF
from engine import Engine
from checkpoint_manager import use_cuda, resume_checkpoint
from torch import nn


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_gmf']
        self.latent_dim_mlp = config['latent_dim_cde']

        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # Параметры для CDECF, включая ode_solver
        config_cdecf = config.copy()
        config_cdecf['latent_dim'] = self.latent_dim_mlp
        config_cdecf['ode_time'] = config.get('ode_time', 1.0)
        config_cdecf['ode_steps'] = config.get('ode_steps', 10)
        config_cdecf['ode_solver'] = config.get('ode_solver', 'dopri5')
        self.cdecf = CDECF(config_cdecf, self.build_dummy_interactions(config_cdecf))

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim_mlp + self.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def build_dummy_interactions(self, config):
        indices = torch.randint(0, config['num_items'], (config['num_users'] * 10,))
        rows = torch.randint(0, config['num_users'], (config['num_users'] * 10,))
        values = torch.ones(config['num_users'] * 10)

        return torch.sparse_coo_tensor(
            torch.stack([rows, indices]),
            values,
            size=(config['num_users'], config['num_items'])
        ).coalesce()

    def forward(self, user_indices, item_indices):
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # MF часть
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # CDECF часть
        mlp_vector = self.cdecf(user_indices, item_indices)

        # Проверка и выравнивание размерностей
        if mf_vector.dim() == 1:
            mf_vector = mf_vector.unsqueeze(-1)

        if mlp_vector.dim() == 1:
            mlp_vector = mlp_vector.unsqueeze(-1)
        elif mlp_vector.dim() == 3:
            mlp_vector = mlp_vector.squeeze(-1)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        config = self.config

        # Загрузка весов GMF
        config_gmf = config.copy()
        config_gmf['latent_dim'] = config_gmf['latent_dim_mf']
        gmf_model = GMF(config_gmf)
        if config['use_cuda']:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_gmf'], device_id=config['device_id'])

        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        # Загрузка весов CDECF
        self.cdecf.load_pretrain_weights()

        # Аффинный слой
        self.affine_output.weight.data = 0.5 * torch.cat(
            [self.cdecf.user_emb.weight.data[:self.num_users], gmf_model.embedding_user.weight.data], dim=-1
        )
        self.affine_output.bias.data.zero_()

class NeuMFEngine(Engine):
    def __init__(self, config):
        self.model = NeuMF(config)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model) # вывод параметров модели

        if config['pretrain']:
            self.model.load_pretrain_weights()

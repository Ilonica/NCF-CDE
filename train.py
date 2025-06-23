import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from ode import CDECFEngine
from neumf import NeuMFEngine
from data import SampleGenerator

def log_metrics(epoch, hr, ndcg, filename='metrics_log.txt'):
    with open(filename, 'a') as f:
        f.write(f'Epoch {epoch}: HR = {hr:.4f}, NDCG = {ndcg:.4f}\n')

gmf_config = {
            'alias': 'gmf_factor8neg4-implict',
            'num_epoch': 75,
            'batch_size': 1024,
            'optimizer': 'adamw',
            'adam_lr': 1e-3,
            'num_users': 6040,
            'num_items': 3706,
            'latent_dim': 8,
            'num_negative': 4,
            'l2_regularization': 0.01,
            'weight_init_gaussian': True,
            'weight_decay': 1e-4,
            'use_cuda': False,
            'use_bachify_eval': False,
            'device_id': 0,
            'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

cde_cf_config = {
            'alias': 'cdecf_factor8neg4',
            'num_epoch': 75,                    # Количество эпох
            'batch_size': 1024,                  # Размер батча
            'optimizer': 'adamw',                # Оптимизатор
            'adam_lr': 5e-4,                     # Скорость обучения
            'num_users': 6040,                   # Кол-во пользователей
            'num_items': 3706,                   # Кол-во фильмов
            'latent_dim': 8,                     # Размерность латентного вектора
            'num_negative': 4,                   # Кол-во негативных примеров
            'layers': [64, 128, 64, 32],         # Уменьшающаяся структура слоев
            'l2_regularization': 0.000001,       # L2-регуляризация
            'weight_init_gaussian': True,        # Инициализация весов нормальным распределением
            'use_cuda': True,                    # Использовать GPU
            'use_bachify_eval': True,            # Использовать батчи при оценке
            'device_id': 0,                      # ID GPU
            'ode_time': 5.0,                     # Интервал времени для ДУ
            'ode_steps': 10,                     # Шаг ODE
            'ode_solver': 'rk4',                 # Метод решения ДУ
            'weight_decay': 1e-4,                # Контролирует значения весов
            'pretrain': True,                    # Предобучение
            'pretrain_gmf': 'checkpoints/gmf_factor8neg4-implict_Epoch74_HR0.6457_NDCG0.3720.model',
            'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

neumf_config = {
    'alias': 'neumf_factor8neg4',
    'num_epoch': 100,                    # Количество эпох
    'batch_size': 1024,                  # Размер батча
    'optimizer': 'adamw',                # Оптимизатор
    'adam_lr': 5e-4,                     # Скорость обучения
    'num_users': 6040,                   # Кол-во пользователей
    'num_items': 3706,                   # Кол-во фильмов
    'latent_dim_gmf': 8,                 # Размерность латентного вектора для GMF
    'latent_dim_cde': 8,                 # Размерность латентного вектора для CDE
    'num_negative': 4,                   # Кол-во негативных примеров
    'layers': [64, 128, 64, 32],         # Уменьшающаяся структура слоев
    'l2_regularization': 0.000001,       # L2-регуляризация
    'weight_init_gaussian': True,        # Инициализация весов нормальным распределением
    'use_cuda': True,                    # Использовать GPU
    'use_bachify_eval': True,            # Использовать батчи при оценке
    'device_id': 0,                      # ID GPU
    'pretrain': True,                    # Предобучение
    'pretrain_gmf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6451_NDCG0.2871.model',
    'pretrain_cde': 'checkpoints/cdecf_factor8neg4_Epoch100_HR0.6527_NDCG0.2955.model',
    'ode_time': 5.0,                     # Интервал времени для ДУ
    'ode_steps': 10,                     # Шаг ODE
    'ode_solver': 'rk4',                 # Метод решения ДУ
    'weight_decay': 1e-4,                # Контроль значения весов
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

# Обработка датасета
ml1m_dir = 'Dataset/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
# config = gmf_config
# engine = GMFEngine(config)
# config = cde_cf_config
# engine = CDECFEngine(config)
config = neumf_config
engine = NeuMFEngine(config)

# Загрузка последнего чекпоинта
last_checkpoint = engine.load_last_checkpoint(config['alias'])
if last_checkpoint:
    start_epoch = last_checkpoint['epoch'] + 1
    print(
        f"Обучение подолжится с эпохи {start_epoch}, HR={last_checkpoint.get('hr', 0):.4f}, NDCG={last_checkpoint.get('ndcg', 0):.4f}")
else:
    start_epoch = 0
    print("Чекпоинты не найдены, обучение начнется с нуля")

# Очистка файла метрик только при начале нового обучения
if start_epoch == 0:
    open('metrics_log.txt', 'w').close()

for epoch in range(start_epoch, config['num_epoch']):
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)

    # Запись метрики в файл
    log_metrics(epoch, hit_ratio, ndcg)










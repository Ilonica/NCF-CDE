import os
import torch
import glob
import numpy as np

# Сохранение чекпоинта
def save_checkpoint(model, model_dir, epoch=None, hr=None, ndcg=None, optimizer=None, config=None):
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'hr': hr,
        'ndcg': ndcg,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config
    }

    torch.save(checkpoint, model_dir)

# Загрузка чекпоинта
def resume_checkpoint(model, model_dir, device_id, optimizer=None):
    map_location = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_dir,
                            map_location=map_location,
                            weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


# Поиск последнего чекпоинта
def find_last_checkpoint(alias, checkpoint_dir='checkpoints'):
    checkpoint_files = glob.glob(f'{checkpoint_dir}/{alias}_Epoch*')
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_Epoch')[1].split('_')[0]))
    return checkpoint_files[-1]

# Конвертация старого формата чекпоинта в новый
def convert_checkpoint(old_path, new_path, model, optimizer=None, config=None):
    old_state = torch.load(old_path)
    new_state = {
        'model_state_dict': old_state,
        'epoch': 0,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config
    }
    torch.save(new_state, new_path)

# Оптимизаторы
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA недостуна'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])

    elif params['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(),
                                      lr=params['adam_lr'],
                                      weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import font
from tkinter import Tk, Label, Button, filedialog
import torch
import pandas as pd
from gmf import GMF
from mlp import MLP
from ode import CDECF
from neumf import NeuMFEngine
from data import SampleGenerator

class MovieRecommenderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Movie Recommender App")
        master.geometry("960x640")
        color = "#dadafe"

        self.movies_df = self.load_movies_data()
        self.itemid_to_title = dict(zip(self.movies_df['itemId'], self.movies_df['title']))
        self.user_ratings = None
        self.user_id = None
        self.model = None
        self.load_button = None

        self.setup_background()
        self.setup_ui()

    # Фоновое изображение
    def setup_background(self):
        try:
            self.master.image = PhotoImage(file='5.png')
            self.bg_logo = Label(self.master, image=self.master.image)
            self.bg_logo.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_logo.lower()
        except Exception as e:
            print(f"Ошибка загрузки фона: {e}")

    def setup_ui(self):
        color = "#dadafe"
        color1 = "#FEFFB2"

        # Очитка старых элементов
        for widget in self.master.winfo_children():
            if widget != self.bg_logo:
                widget.destroy()

        system_label = Label(self.master, text="Рекомендательная система фильмов",
                             font=("Helvetica", 20, "bold"), bg=color)
        system_label.place(relx=0.5, rely=0.07, anchor="center")

        self.system_label1 = Label(self.master,
                                   text="С наибольшей вероятностью Вам понравятся следующие фильмы:",
                                   font=("Helvetica", 15), bg=color)

        # Выбор метода
        self.method_var = StringVar(value="NCF")
        self.checkbox_ncf = Radiobutton(self.master, text="Составить рекомендацию на основе метода NCF",
                                        variable=self.method_var, value="NCF",
                                        font=("Helvetica", 15), bg=color)
        self.checkbox_ncf.place(relx=0.5, rely=0.2, anchor="center")

        self.checkbox_ncf_cde = Radiobutton(self.master, text="Составить рекомендацию на основе метода NCF-CDE",
                                            variable=self.method_var, value="NCF-CDE",
                                            font=("Helvetica", 15), bg=color)
        self.checkbox_ncf_cde.place(relx=0.5, rely=0.25, anchor="center")

        # Кнопка загрузки
        custom_font = font.Font(family="Helvetica", size=15, weight="bold")
        self.load_button = Button(self.master, text="Загрузить оценки", command=self.load_ratings,
                                  width=30, height=2, bg=color1, relief="ridge", bd=5, font=custom_font)
        self.load_button.place(relx=0.5, rely=0.85, anchor="center")

        self.result_label = Label(self.master, text="", font=("Helvetica", 17), bg=color)
        self.result_label.place(relx=0.5, rely=0.51, anchor="center")

    # Сброс UI
    def reset_application(self):
        self.user_ratings = None
        self.user_id = None
        self.model = None
        self.setup_ui()

    # Замена текста кнопки на "Очистить"
    def change_to_clear_button(self):

        self.load_button.config(text="Очистить", command=self.reset_application)

    def hide_method_selector(self):
        self.checkbox_ncf.place_forget()
        self.checkbox_ncf_cde.place_forget()

    def get_model_config(self):
        if self.method_var.get() == "NCF":
            return {
                'alias': 'neumf_factor8neg4',
                'num_epoch': 100,
                'batch_size': 1024,
                'optimizer': 'adamw',
                'adam_lr': 3e-4,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_gmf': 8,
                'latent_dim_cde': 8,
                'num_negative': 4,
                'layers': [16, 64, 16, 8],
                'l2_regularization': 0.000001,
                'weight_init_gaussian': True,
                'use_cuda': False,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model',
                'pretrain_mlp': 'checkpoints/mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model',
                'weight_decay': 1e-4,
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
            }
        else:
            return {
                'alias': 'neumf_factor16neg10',
                'num_epoch': 100,
                'batch_size': 1024,
                'optimizer': 'adamw',
                'adam_lr': 5e-4,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_gmf': 8,
                'latent_dim_cde': 8,
                'num_negative': 4,
                'layers': [64, 128, 64, 32],
                'l2_regularization': 0.000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_gmf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6451_NDCG0.2871.model',
                'pretrain_cde': 'checkpoints/cdecf_factor8neg4_Epoch100_HR0.6527_NDCG0.2955.model',
                'ode_time': 5.0,
                'ode_steps': 10,
                'ode_solver': 'rk4',
                'weight_decay': 1e-4,
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
            }

    def get_model_weights_path(self):
        if self.method_var.get() == "NCF":
            return 'checkpoints/neumf_factor8neg4_Epoch99_HR0.6906_NDCG0.4152.model'
        else:
            return 'checkpoints/neumf_factor16neg10_Epoch99_HR0.6858_NDCG0.4134.model'

    # Загрузка модели в зависимости от метода
    def load_model(self):
        neumf_config = self.get_model_config()
        model_path = self.get_model_weights_path()

        try:
            engine = NeuMFEngine(neumf_config)

            # Загрузка checkpoint
            checkpoint = torch.load(
                model_path,
                map_location='cuda' if neumf_config['use_cuda'] else 'cpu',
                weights_only=False
            )

            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # Загружаем только веса
            engine.model.load_state_dict(state_dict)

            if neumf_config['use_cuda']:
                engine.model.cuda()

            engine.model.eval()
            print(f"Модель {self.method_var.get()} успешно загружена")
            return engine

        except Exception as e:
            print(f"Ошибка загрузки модели {self.method_var.get()}: {e}")
            return None

    def load_movies_data(self):
        try:
            return pd.read_csv('Dataset/movies.csv')
        except:
            return pd.read_csv('Dataset/movies.dat', sep='::', engine='python',
                             names=['itemId', 'title', 'genre'])

    def load_ratings(self):
        file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
        if not file_path:
            return

        try:
            # Загрузка оценок пользователя
            ratings = pd.read_csv(file_path, sep='::',
                                  names=['userId', 'itemId', 'rating', 'timestamp'],
                                  engine='python')

            if len(ratings) == 0:
                raise ValueError("Файл с оценками пуст")

            self.user_id = ratings['userId'].iloc[0]
            self.user_ratings = ratings

            # Объединяем данные пользователей
            try:
                ml1m_rating = pd.read_csv('Dataset/ratings.dat', sep='::',
                                          names=['userId', 'itemId', 'rating', 'timestamp'],
                                          engine='python')
                all_ratings = pd.concat([ml1m_rating, ratings])
            except:
                all_ratings = ratings

            self.sample_generator = SampleGenerator(ratings=all_ratings)
            self.model = self.load_model()  # Загрузка выбранной модели

            self.hide_method_selector()

            self.change_to_clear_button()

            self.system_label1.place(relx=0.5, rely=0.15, anchor="center")
            self.get_recommendations()

        except Exception as e:
            self.result_label.config(text=f"Ошибка загрузки файла: {str(e)}")

    # Составление топ-10 рекомендаций для пользователя
    def get_top_n_recommendations(self, user_id, model, sample_generator, n=10):
        try:
            # Извлечение всех фильмов, которые пользователь еще не оценивал
            rated_items = set(self.user_ratings['itemId']) if self.user_ratings is not None else set()

            # Фильтруем только существующие itemId
            all_items = [item for item in sample_generator.item_pool
                         if item not in rated_items and item in self.itemid_to_title]

            if not all_items:
                print("Нет доступных фильмов для рекомендаций")
                return []

            user_tensor = torch.LongTensor([user_id] * len(all_items))
            item_tensor = torch.LongTensor(all_items)

            if model.model.config.get('use_cuda', False):
                user_tensor = user_tensor.cuda()
                item_tensor = item_tensor.cuda()

            with torch.no_grad():
                predictions = model.model(user_tensor, item_tensor)

            if predictions.numel() == 0:
                print("Модель не вернула предсказания")
                return []

            # Получаем топ-10 рекомендаций
            k = min(n, len(all_items))
            _, top_indices = torch.topk(predictions.flatten(), k=k)

            # Проверка индексов на валидность
            valid_indices = [i.item() for i in top_indices if i < len(all_items)]
            recommendations = [all_items[i] for i in valid_indices]

            print(f"Найдено {len(recommendations)} рекомендаций")
            return recommendations

        except Exception as e:
            print(f"Ошибка в get_top_n_recommendations: {e}\n{traceback.format_exc()}")
            return []

    def get_recommendations(self):
        if not self.user_id or not self.model:
            self.result_label.config(text="Ошибка: не загружены данные пользователя или модель")
            return

        try:
            top_items = self.get_top_n_recommendations(self.user_id, self.model, self.sample_generator)

            # Проверка
            print(f"Получено {len(top_items)} рекомендаций")
            print("ID рекомендованных фильмов:", top_items)

            if not top_items:
                self.result_label.config(
                    text="Не удалось получить рекомендации.")
                return

            recommendations_text = f"Топ-10 рекомендаций ({self.method_var.get()}):\n"
            for i, item in enumerate(top_items, 1):
                title = self.itemid_to_title.get(item, f"Фильм с ID {item}")
                print(f"{i}. {title} (ID: {item})")
                recommendations_text += f"{i}. {title}\n"

            self.result_label.config(text=recommendations_text)

        except Exception as e:
            print(f"Полная ошибка: {traceback.format_exc()}")
            self.result_label.config(text=f"Ошибка при получении рекомендаций: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    root.image = PhotoImage(file='5.png')
    bg_logo = Label(root, image=root.image)
    bg_logo.place(x=0, y=0, relwidth=1, relheight=1)
    app = MovieRecommenderApp(root)
    root.mainloop()
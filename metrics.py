import math
import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]

        # Логирование размеров данных (для отладки)
        print(f"Length of test_users: {len(test_users)}")
        print(f"Length of test_items: {len(test_items)}")
        print(f"Length of test_preds: {len(test_scores)}")

        # Создание DataFrame с целевыми (тестовыми) парами
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})

        # Создание общего DataFrame со всеми парами (целевые + негативные)
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        # Объединение с целевыми парами для каждого пользователя
        full = pd.merge(full, test, on=['user'], how='left')
        # Ранжирование товаров по предсказанным рейтингам для каждого пользователя
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    # Метрика HitRatio
    def cal_hit_ratio(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k] # Находим фильмы, попавшие в топ-K
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    # Метрика NDCG
    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
        test_in_top_k.loc[:, 'ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()


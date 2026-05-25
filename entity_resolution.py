#!/usr/bin/env python
# coding: utf-8

# # Imports + EDA

# In[1]:


# --- ОСНОВНЫЕ БИБЛИОТЕКИ ---
import pandas as pd
import numpy as np
import json
import joblib
import os
import re
import random
from collections import defaultdict
from itertools import combinations
import time
from pathlib import Path

# --- СТРОКОВЫЕ МЕТРИКИ ---
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

# --- МАТЕМАТИКА И ГРАФЫ ---
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- ML ---
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, roc_auc_score, 
                              confusion_matrix, precision_recall_curve)
import lightgbm as lgb

# --- ПОИСК И ЭМБЕДДИНГИ ---
import faiss
from sentence_transformers import SentenceTransformer

# --- ВИЗУАЛИЗАЦИЯ ---
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("Все библиотеки успешно импортированы!")


# In[2]:


df = pd.read_parquet('../student_dataset/student_relations.parquet')

print("Размер:", df.shape)
print("\nТипы колонок:")
print(df.dtypes)
print("\nПустые значения:")
print(df.isnull().sum())
print("\nСтраны:")
print(df['country'].value_counts())


# In[3]:


for country in df['country'].unique():
    print(f"\n=== {country} ===")
    sample = df[df['country'] == country]['party_name'].dropna().head(5)
    print(sample.values)


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


print("Уникальных party_name:", df['party_name'].nunique())
print("Всего строк:", len(df))
print("\nСамые частые имена:")
print(df['party_name'].value_counts().head(20))


# In[7]:


# Смотрим на relation_kind
print(df['relation_kind'].value_counts())

# Смотрим на party_type  
print(df['party_type'].value_counts())


# In[8]:


# Смотрим примеры individual vs legal_entity
print("=== individual ===")
print(df[df['party_type']=='individual']['party_name'].head(10).values)

print("\n=== legal_entity ===")
print(df[df['party_type']=='legal_entity']['party_name'].head(10).values)


# In[9]:


print("Пустых company_name_norm:", df['company_name_norm'].isnull().sum())
print("\nПримеры по странам:")
for country in df['country'].unique():
    print(f"\n=== {country} ===")
    sample = df[df['country']==country]['company_name_norm'].dropna().head(3)
    print(sample.values)


# In[10]:


print(df['company_name_norm'].value_counts().head(20))


# In[11]:


# Есть ли случаи где один party_public_id встречается много раз?
id_counts = df[df['party_public_id'].notna()]['party_public_id'].value_counts()
print("Топ party_public_id по количеству записей:")
print(id_counts.head(10))
print("\nСредние количество записей на один id:", id_counts.mean().round(2))


# In[12]:


top_id = 'arm_party_08c10e0c2e663101'
print(df[df['party_public_id'] == top_id]['party_name'].value_counts().head(5))


# In[13]:


# Сколько записей с party_public_id
has_id = df[df['party_public_id'].notna()]
print("Записей с id:", len(has_id))
print("Уникальных id:", has_id['party_public_id'].nunique())
print("\nРаспределение размеров групп:")
print(has_id['party_public_id'].value_counts().describe())


# In[14]:


has_id['party_public_id'].value_counts()


# In[15]:


vc = has_id['party_public_id'].value_counts()
vc[vc == 2]


# In[16]:


has_id[has_id.party_public_id == 'arm_party_08c10e0c2e663101']['company_name_norm'].unique()


# In[17]:


has_id[has_id.party_public_id == 'tjk_party_49a1ded4d83ff467']


# In[18]:


# Считаем пары внутри каждой группы
groups = has_id.groupby('party_public_id').size()
# убираем мусорные большие группы
groups_clean = groups[(groups > 1) & (groups < 100)]
pairs_count = (groups_clean * (groups_clean - 1) / 2).sum()
print("Групп с дубликатами:", len(groups_clean))
print("Позитивных пар (label=1):", int(pairs_count))


# # Разметка

# In[19]:


def preprocess_name(name):
    if not isinstance(name, str):
        return ""

    # нижний регистр
    name = name.lower()

    # убираем кавычки и спецсимволы
    name = re.sub(r'[«»\"\'<>()]', '', name)

    # убираем лишние пробелы
    name = re.sub(r'\s+', ' ', name).strip()

    return name

# проверяем
examples = [
    "ИВАНОВ ИВАН ИВАНОВИЧ",
    "«БЛОК-СТОН»",
    "Замените на физическое лицо 1",
    "Անհայտ կազմakerpутюн",
]

for e in examples:
    print(f"{e} → {preprocess_name(e)}")


# In[20]:


# Список мусорных паттернов которые нашли в EDA
JUNK_PATTERNS = [
    'замените на физическое лицо',
    'անհայտ կազмakerpутюн',  
    'անhայт',
    'неизвестн',
]

def is_junk(name):
    if not isinstance(name, str):
        return True
    name_lower = name.lower()
    for pattern in JUNK_PATTERNS:
        if pattern in name_lower:
            return True
    return False

# проверяем
examples = [
    "Замените на физическое лицо 1",
    "Иванов Иван Иванович",
    "Անhայт կազмakerpутюн",
    "Neo Metals Holdings",
]

for e in examples:
    print(f"{e} → мусор: {is_junk(e)}")


# In[21]:


top_names = df['party_name'].value_counts().head(50)
print(top_names)


# In[22]:


df['is_junk'] = df['party_name'].apply(is_junk)
print("Мусорных записей:", df['is_junk'].sum())
print("Процент:", (df['is_junk'].sum() / len(df) * 100).round(2), "%")


# In[23]:


# убираем мусор
df_clean = df[~df['is_junk']].copy()

# применяем preprocess_name к именам
df_clean['party_name_clean'] = df_clean['party_name'].apply(preprocess_name)
df_clean['company_name_clean'] = df_clean['company_name_norm'].apply(preprocess_name)

print("Было строк:", len(df))
print("Стало строк:", len(df_clean))
print("\nПример:")
print(df_clean[['party_name', 'party_name_clean']].head(5))


# In[24]:


# Посмотрим на объемы уникальных компаний
print("Уникальных нормализованных компаний:", df_clean['company_name_norm'].nunique())

# Проверим, есть ли компании, которые одновременно побывали в роли связанных лиц (party_name)
companies = set(df_clean['company_name_norm'].dropna().unique())
parties = set(df_clean['party_name_clean'].dropna().unique())

intersection = companies.intersection(parties)
print(f"Найдено {len(intersection)} сущностей, которые встречаются и как компании, и как связанные лица!")


# In[25]:


# берём только записи с company_public_id
df_labeled = df_clean[df_clean['company_public_id'].notna()].copy()

# считаем сколько УНИКАЛЬНЫХ имён у каждого id
group_sizes = df_labeled.groupby('company_public_id')['company_name_clean'].nunique()

# оставляем группы где больше 1 уникального имени
valid_ids = group_sizes[(group_sizes > 1) & (group_sizes < 100)].index
df_labeled = df_labeled[df_labeled['company_public_id'].isin(valid_ids)]

print("Записей для формирования пар:", len(df_labeled))
print("Уникальных id:", df_labeled['company_public_id'].nunique())


# In[26]:


random.seed(42)

print("--- Обработка компаний ---")

# 2. Позитивные пары компаний (внутри одного ID)
pos_pairs = []
for comp_id, group in df_labeled.groupby('company_public_id'):
    unique_names = group[['company_name_clean', 'country']].drop_duplicates().to_dict('records')
    if len(unique_names) > 1:
        for pair in combinations(unique_names, 2):
            pos_pairs.append({
                'name_a': pair[0]['company_name_clean'], 'country_a': pair[0]['country'],
                'name_b': pair[1]['company_name_clean'], 'country_b': pair[1]['country'],
                'label': 1
            })

df_pos_companies = pd.DataFrame(pos_pairs)
print(f"Позитивных пар компаний сформировано: {len(df_pos_companies)}")

# 3. Негативные пары компаний
neg_pairs = []
all_companies = df_labeled[['company_name_clean', 'country', 'company_public_id']].drop_duplicates().to_dict('records')

for row in all_companies:
    if random.random() < 0.7:
        candidates = [c for c in all_companies 
                      if c['company_public_id'] != row['company_public_id'] and c['country'] == row['country']]
    else:
        candidates = [c for c in all_companies 
                      if c['company_public_id'] != row['company_public_id'] and c['country'] != row['country']]

    if candidates:
        enemy = random.choice(candidates)
        neg_pairs.append({
            'name_a': row['company_name_clean'], 'country_a': row['country'],
            'name_b': enemy['company_name_clean'], 'country_b': enemy['country'],
            'label': 0
        })

    if len(neg_pairs) >= len(df_pos_companies):
        break

df_neg_companies = pd.DataFrame(neg_pairs)
print(f"Негативных пар компаний сформировано: {len(df_neg_companies)}")


# In[27]:


random.seed(42)

print("--- Обработка людей ---")

def augment_name(name):
    if not isinstance(name, str) or len(name) < 3:
        return name
    chars = list(name)
    aug_type = random.choice(['typo', 'swap', 'drop'])
    if aug_type == 'typo':
        idx = random.randint(1, len(chars)-1)
        chars[idx] = random.choice('абвгдеёжзийклмнопрстуфхцчшщыьэюя')
    elif aug_type == 'swap' and len(chars) > 2:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif aug_type == 'drop' and len(chars) > 3:
        idx = random.randint(1, len(chars)-1)
        chars.pop(idx)
    return ''.join(chars)

# 1. Уникальные люди
people = df_clean[df_clean['party_type'] == 'individual']['party_name_clean'].dropna().unique().tolist()

# 2. Позитивные пары (опечатки)
aug_pos = []
for name in random.sample(people, min(20000, len(people))):
    aug = augment_name(name)
    if aug != name:
        aug_pos.append({
            'name_a': name, 'name_b': aug, 
            'country_a': 'unknown', 'country_b': 'unknown', 
            'label': 1
        })

# 3. Сложные негативные пары (однофамильцы)
surname_groups = defaultdict(list)
for name in people:
    words = name.split()
    if len(words) > 0:
        surname_groups[words[0]].append(name)

aug_neg = []
surnames = list(surname_groups.keys())
random.shuffle(surnames)

for surname in surnames:
    group = surname_groups[surname]
    if len(group) > 1:
        for i in range(len(group) - 1):
            name_a = group[i]
            name_b = group[i+1]
            if name_a != name_b:
                aug_neg.append({
                    'name_a': name_a, 'name_b': name_b, 
                    'country_a': 'unknown', 'country_b': 'unknown', 
                    'label': 0
                })
            if len(aug_neg) >= len(aug_pos):
                break
    if len(aug_neg) >= len(aug_pos):
        break

# Страховочный добор случайных людей
if len(aug_neg) < len(aug_pos):
    needed = len(aug_pos) - len(aug_neg)
    sample_people = random.sample(people, min(needed * 2, len(people)))
    for i in range(0, len(sample_people)-1, 2):
        aug_neg.append({
            'name_a': sample_people[i], 'name_b': sample_people[i+1], 
            'country_a': 'unknown', 'country_b': 'unknown', 
            'label': 0
        })

df_aug = pd.DataFrame(aug_pos + aug_neg)
print(f"Сгенерировано для людей: Позитивных - {len(aug_pos)}, Негативных (Hard) - {len(aug_neg)}")


# In[28]:


# Склеиваем строго свежесозданные изолированные переменные
df_train = pd.concat([df_pos_companies, df_neg_companies, df_aug], ignore_index=True)

# Перемешиваем строки, чтобы бустинг учился равномерно
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Итоговый размер общего датасета: {len(df_train)} строк")
print("Распределение классов в таргете:")
print(df_train['label'].value_counts())


# # Feature Engineering

# In[29]:


def compute_features(row):
    a = row['name_a']
    b = row['name_b']

    # Исправление логики стран: если страны неизвестны (это люди), ставим 0
    if row['country_a'] == 'unknown' or row['country_b'] == 'unknown':
        same_country = 0
    else:
        same_country = int(row['country_a'] == row['country_b'])

    return pd.Series({
        'jaro_winkler': JaroWinkler.similarity(a, b),
        'levenshtein': fuzz.ratio(a, b) / 100,
        'token_sort': fuzz.token_sort_ratio(a, b) / 100,
        'partial': fuzz.partial_ratio(a, b) / 100,
        'same_country': same_country,
        'len_diff': abs(len(a) - len(b)) / max(len(a), len(b), 1),
    })


# In[30]:


# start = time.time()
# df_features = df_train.apply(compute_features, axis=1)
# df_features['label'] = df_train['label'].values
# end = time.time()

# print(f"Время: {end-start:.1f} сек")
# print(f"Строк: {len(df_features)}")
# print(df_features.describe())


# In[48]:


# 1. Загружаем модель
model_emb = SentenceTransformer('model')

names_a = df_train['name_a'].astype(str).tolist()
names_b = df_train['name_b'].astype(str).tolist()

features_file = Path("features.csv")

if features_file.is_file():
    df_features = pd.read_csv('features.csv')
else:
    df_features = df_train.apply(compute_features, axis=1)
    df_features['label'] = df_train['label'].values

    print("Кодируем name_a...")
    emb_a = model_emb.encode(names_a, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    print("Кодируем name_b...")
    emb_b = model_emb.encode(names_b, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    # 2. Быстрая нормализация векторов (чтобы dot product стал равен cosine similarity)
    emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    emb_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)

    print("Считаем косинусное сходство по строкам (матрично)...")
    # Умножаем поэлементно векторы из каждой пары и суммируем по строкам
    cosine_scores = np.sum(emb_a * emb_b, axis=1)

    # Добавляем в наш датасет признаков
    df_features['cosine_sim'] = cosine_scores
    df_features.to_csv('features.csv', index=False)

print("Все фичи готовы! Проверяем форму:", df_features.shape)
print(df_features[['jaro_winkler', 'cosine_sim', 'label']].head(5))


# # Model

# In[49]:


# 1. Выделяем признаки (фичи) и целевую переменную (таргет)
X = df_features.drop('label', axis=1)
y = df_features['label']

# 2. Разбиваем на train/test (20%) со стратификацией для сохранения баланса классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Инициализируем базовый классификатор LightGBM с оптимальными параметрами
base_model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=250, 
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    verbose=-1  # отключаем лишние технические логи LightGBM
)

print("Шаг 1: Обучение базовой модели и изотоническая калибровка вероятностей...")
# 4. Калибровка модели методом Isotonic Regression через встроенную кросс-валидацию.
# Это гарантирует, что предсказанные скоры будут отражать честную математическую вероятность дублирования,
# что критически важно для построения корректного графа связности.
model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
model.fit(X_train, y_train)

# 5. Получаем откалиброванные вероятности на тестовой выборке
y_prob = model.predict_proba(X_test)[:, 1]

print("Шаг 2: Вычисление Precision-Recall кривой и автоматический подбор порога...")
# 6. Строим Precision-Recall кривую для анализа всех возможных порогов отсечения
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Задаем коэффициент бета для F-beta score.
# Мы сознательно выбираем beta = 0.5 (метрика F0.5), так как в задаче дедупликации 3.6 млн строк 
# точность (Precision) в два раза важнее полноты (Recall). Ложная склейка тёзок или разных компаний 
# для бизнеса критичнее, чем пара ненайденных дубликатов.
beta = 0.5
beta_sq = beta ** 2

# Вычисляем F_0.5 score для каждой точки кривой
f_beta_scores = (1 + beta_sq) * (precisions * recalls) / ((beta_sq * precisions) + recalls + 1e-10)

# Находим индекс максимального значения и извлекаем соответствующий ему порог
best_idx = np.argmax(f_beta_scores)
best_threshold = thresholds[best_idx]

print(f"\n🎯 Математически оптимальный порог по F_{beta}-score: {best_threshold:.4f}")

# 7. Применяем наш кастомный строгий порог к полученным вероятностям
y_pred_custom = (y_prob >= best_threshold).astype(int)

# 8. Выводим итоговые метрики качества
print("\n=== ИТОГОВЫЕ МЕТРИКИ КАЧЕСТВА (ПОСЛЕ КАЛИБРОВКИ И ОПТИМИЗАЦИИ ПОРОГА) ===")
print(classification_report(y_test, y_pred_custom, target_names=['Не дубликат', 'Дубликат']))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# Перезаписываем глобальную переменную y_pred, чтобы следующая ячейка матрицы ошибок 
# и анализа ложных позитивов/негативов автоматически подхватила новые, оптимизированные данные
y_pred = y_pred_custom


# In[50]:


# У CalibratedClassifierCV обученные копии базовой модели лежат в списке calibrated_classifiers_
fitted_base_model = model.calibrated_classifiers_[0].estimator

lgb.plot_importance(fitted_base_model, max_num_features=10, figsize=(10, 6))
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# In[51]:


# кодируем все уникальные имена из датасета
print("Кодируем все имена...")
# Изменяем tolist() на np.array, чтобы индексный маппинг all_names[j] работал корректно во всех ячейках
all_names = np.array(df_clean['party_name_clean'].dropna().unique())
print(f"Всего уникальных имён: {len(all_names)}")

faiss_index_file = Path("faiss_index.bin")

if faiss_index_file.is_file():
    index = faiss.read_index('faiss_index.bin')
else:
    embeddings = model_emb.encode(all_names.tolist(), batch_size=512, show_progress_bar=True)
    embeddings = embeddings.astype('float32')

    # нормализуем для cosine similarity
    faiss.normalize_L2(embeddings)

    # строим индекс
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, 'faiss_index.bin')

print(f"Индекс построен! Векторов: {index.ntotal}")


# In[52]:


def search_duplicates(query_name, top_k=10):
    # 1. Получаем эмбеддинг запроса
    query_clean = preprocess_name(query_name)
    query_emb = model_emb.encode([query_clean], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)

    # 2. FAISS достаёт топ-50 кандидатов
    D, I = index.search(query_emb, 50)
    candidates = [all_names[i] for i in I[0]]

    # 3. Собираем структуру для батч-вычисления фич
    raw_pairs = []
    for cand in candidates:
        raw_pairs.append({
            'name_a': query_clean,
            'name_b': cand,
            'country_a': 'unknown',
            'country_b': 'unknown'
        })
    df_pairs = pd.DataFrame(raw_pairs)

    # Применяем compute_features одной пачкой по строкам
    df_cands = df_pairs.apply(compute_features, axis=1)
    df_cands['cosine_sim'] = D[0].astype(float)

    # Явно задаем порядок колонок, строго как при обучении!
    feature_cols = ['jaro_winkler', 'levenshtein', 'token_sort', 'partial', 
                    'same_country',
                    'len_diff', 'cosine_sim']
    X_matrix = df_cands[feature_cols]

    # 4. Откалиброванный LightGBM скорит кандидатов по правильной матрице
    scores = model.predict_proba(X_matrix)[:, 1]

    # Собираем красивый финальный датафрейм для вывода
    df_res = pd.DataFrame({
        'candidate': candidates,
        'score': scores
    })

    return df_res.nlargest(top_k, 'score').reset_index(drop=True)

# Тест
print(search_duplicates("мельников андрей"))


# In[53]:


# print(search_duplicates("нео металс холдинг"))
# print("---")
# print(search_duplicates("сбербанк"))


from fastapi import FastAPI, HTTPException

app = FastAPI(title="Entity Resolution API")

@app.post("/search")
def search(entity: str):

    return search_duplicates("entity").to_json()

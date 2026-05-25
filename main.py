import re
import json
import pickle
import faiss
import os
import numpy as np
from rapidfuzz import distance, fuzz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

assets = {'model': None, 'index': None, 'names': None, 'embedder': None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = {
        'model': 'model_lgbm.pkl',
        'index': 'faiss_index.bin',
        'names': 'index_names.json',
        'embedder_dir': 'model/'
    }

    print(f"[LIFESPAN] Checking for embedder_dir: {paths['embedder_dir']}")
    if os.path.exists(paths['embedder_dir']):
        try:
            assets['embedder'] = SentenceTransformer(paths['embedder_dir'])
            print('[LIFESPAN] SentenceTransformer loaded.')
        except Exception as e: 
            print(f'[LIFESPAN] Error loading BERT: {e}')
    else:
        print(f'[LIFESPAN] Embedder directory not found: {paths["embedder_dir"]}')

    print(f"[LIFESPAN] Checking for model file: {paths['model']}")
    if os.path.exists(paths['model']):
        try:
            with open(paths['model'], 'rb') as f: assets['model'] = pickle.load(f)
            print('[LIFESPAN] LGBM loaded.')
        except Exception as e: 
            print(f'[LIFESPAN] Error loading LGBM: {e}')
    else:
        print(f'[LIFESPAN] LGBM model file not found: {paths["model"]}')

    print(f"[LIFESPAN] Checking for names file: {paths['names']}")
    if os.path.exists(paths['names']):
        try:
            with open(paths['names'], 'r') as f: assets['names'] = json.load(f)
            print('[LIFESPAN] Names loaded.')
        except Exception as e: 
            print(f'[LIFESPAN] Error loading names: {e}')
    else:
        print(f'[LIFESPAN] Names file not found: {paths["names"]}')

    print(f"[LIFESPAN] Checking for FAISS index file: {paths['index']}")
    if os.path.exists(paths['index']):
        try:
            assets['index'] = faiss.read_index(paths['index'])
            print('[LIFESPAN] FAISS index loaded.')
        except Exception as e: 
            print(f'[LIFESPAN] Error loading FAISS: {e}')
    else:
        print(f'[LIFESPAN] FAISS index file not found: {paths["index"]}')

    print(f"[LIFESPAN] Assets after loading: {{'model': {assets['model'] is not None}, 'index': {assets['index'] is not None}, 'names': {assets['names'] is not None}, 'embedder': {assets['embedder'] is not None}}}")

    yield
    assets.clear()

app = FastAPI(title='Entity Resolution API', lifespan=lifespan)

def preprocess_name(name):
    if not isinstance(name, str): return ''
    name = name.lower()
    name = re.sub(r'[«»"''<>()]', '', name)
    return re.sub(r'\s+', ' ', name).strip()

def compute_features(name_a, name_b):
    return [
        fuzz.ratio(name_a, name_b) / 100.0,
        distance.Levenshtein.distance(name_a, name_b),
        1.0 if name_a and name_b and name_a[0] == name_b[0] else 0.0
    ]

class CheckRequest(BaseModel):
    name_a: str
    name_b: str

@app.post('/search')
def search(name: str, top_k: int = 10):
    if not assets['index'] or not assets['embedder']:
        raise HTTPException(status_code=503, detail='Search unavailable')
    query_vec = assets['embedder'].encode([preprocess_name(name)]).astype('float32')
    distances, indices = assets['index'].search(query_vec, top_k)
    results = []
    for d, idx in zip(distances[0], indices[0]):
        if idx < len(assets['names']): results.append({'name': assets['names'][idx], 'score': float(d)})
    return results

@app.post('/check_duplicate')
def check_duplicate(req: CheckRequest):
    if not assets['model']: raise HTTPException(status_code=503, detail='Model not loaded')
    # Corrected bug: req.b -> req.name_b
    feats = compute_features(preprocess_name(req.name_a), preprocess_name(req.name_b))
    prob = assets['model'].predict_proba([feats])[0][1]
    return {'probability': float(prob), 'is_duplicate': bool(prob > 0.5)}

@app.get('/health')
def health():
    return {'status': 'ok', 'loaded': {k: (v is not None) for k, v in assets.items()}}
FROM python:3.11-bookworm
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY faiss_index.bin ./
COPY model_lgbm.pkl ./
COPY features.csv ./
COPY entity_resolution.py ./
EXPOSE 8000
CMD [ "fastapi", "run", "entity_resolution.py" ]

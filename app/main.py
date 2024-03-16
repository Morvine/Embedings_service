import io
import json
import os
import minio
import pickle
import torch
from fastapi import FastAPI, Body, status, HTTPException
from fastapi.responses import JSONResponse

import hashlib


from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from app.logic.create_emb import create_text, build_index
from app.logic.retrieve_utils import retrieve


load_dotenv()


model_path = f"{os.getenv('MODELS_PATH', '/volumes/ml_models')}/{os.getenv('MODEL_NAME', 'e5-large-en-ru')}"
model = SentenceTransformer(model_path)

app = FastAPI()
client = minio.Minio(
    endpoint=os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    access_key=os.getenv('MINIO_ACCESS_KEY', 'cjKMqPAaGfpnsIdRzNZG'),
    secret_key=os.getenv('MINIO_SECRET_KEY', 'WjNFiKfpZAVBDScjhp6w4KzFSy7jRkuB50EhoVl3'),
    secure=False
)


MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'my-bucket')

CHUNK_SIZE = eval(os.getenv('CHUNK_SIZE', 400))
CHUNK_OVERLAP = eval(os.getenv('CHUNK_SIZE', 45))
E5_FLAG = eval(os.getenv('E5_FLAG', True))
ADD_SPEAKER = eval(os.getenv('ADD_SPEAKER', True))

@app.post("/create_embeddings")
def create_embeddings(data=Body()):
    minio_path = data['file_path']
    base_path = os.path.dirname(minio_path)

    try:
        response = client.get_object(MINIO_BUCKET, minio_path)
        data = response.data.decode('utf8').replace("'", '"')
        data = json.loads(data)
    except ValueError:
        return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "Файл не найден"}
        )
    finally:
        response.close()
        response.release_conn()

    data_text = create_text(data, add_speaker=ADD_SPEAKER)
    db = build_index(model, data_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_SIZE, e5_flag=E5_FLAG)
    if db is not None and db['docs'] is not None and db['embeddings'] is not None:
        value_as_bytes = pickle.dumps(db)
        res_filename = f'{base_path}/' + str(hashlib.md5(value_as_bytes).hexdigest()) + '.pickle'

        res_data = io.BytesIO(value_as_bytes)

        client.put_object(bucket_name=MINIO_BUCKET, object_name=f'{res_filename}', data=res_data,
                          length=len(value_as_bytes))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'result': res_filename})

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Отсутствует результат декодирования"})



@app.post("/retrieve_docs")
def retrieve_docs(data=Body()):
    minio_path = data['file_path']
    question = data['question']
    base_path = os.path.dirname(minio_path)

    try:
        response = client.get_object(MINIO_BUCKET, minio_path)
        data = pickle.loads(response.data)
    except ValueError:
        return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "Файл не найден"}
        )
    finally:
        response.close()
        response.release_conn()

    retrieved_docs = retrieve(model, question, data, k_documents=7, e5_flag=E5_FLAG, beta=0.3, alpha=0.1)
    result = {'data': retrieved_docs}

    if result is not None and result['data']:
        value_as_bytes = str(result).encode('utf-8')
        res_filename = f'{base_path}/' + str(hashlib.md5(value_as_bytes).hexdigest()) + '.json'

        res_data = io.BytesIO(value_as_bytes)

        client.put_object(bucket_name=MINIO_BUCKET, object_name=f'{res_filename}', data=res_data,
                          length=len(value_as_bytes))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'result': res_filename})

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Отсутствует результат декодирования"})




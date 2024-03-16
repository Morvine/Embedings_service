from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_text(data, add_speaker=True):
    data_text = ''
    for d in data['data']:
        if add_speaker:
            data_text += f"{d['speaker']}\n"
        data_text += f"{d['text']}\n"
        if not add_speaker:
            data_text += "\n"
    return data_text



def build_index(EMBEDDER, data_text, chunk_size=250, chunk_overlap=30, e5_flag=False):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(data_text)
    if e5_flag:
        for i in range(len(texts)):
            texts[i] = f'passage: {texts[i]}'
    if e5_flag:
        embeddings = EMBEDDER.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    else:
        embeddings = EMBEDDER.encode(texts, convert_to_tensor=True)
    db = {"docs": texts, "embeddings": embeddings}
    return db
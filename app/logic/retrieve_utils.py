import torch
from sentence_transformers.util import cos_sim

def fill_dyn_matrix(x, y):
    L = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for x_i,x_elem in enumerate(x):
        for y_i,y_elem in enumerate(y):
            if x_elem == y_elem:
                L[x_i][y_i] = L[x_i-1][y_i-1] + 1
            else:
                L[x_i][y_i] = max((L[x_i][y_i-1],L[x_i-1][y_i]))
    return L

def LCS_DYN(x, y):
    L = fill_dyn_matrix(x, y)
    LCS = []
    x_i,y_i = len(x)-1,len(y)-1
    while x_i >= 0 and y_i >= 0:
        if x[x_i] == y[y_i]:
            LCS.append(x[x_i])
            x_i, y_i = x_i-1, y_i-1
        elif L[x_i-1][y_i] > L[x_i][y_i-1]:
            x_i -= 1
        else:
            y_i -= 1
    LCS.reverse()
    return LCS

def calc_lcs(question, data, beta=1.5):
    result = []
    for tpk in data:
        a = len(LCS_DYN(question, tpk))/len(tpk)
        b = len(LCS_DYN(question, tpk))/len(question)
        f = (1+beta)*a*b/(a+beta*b + 1e-7)
        result.append(f)
    result = torch.tensor(result)
    return result


def retrieve(EMBEDDER, question, db, k_documents=8, k_first=None, e5_flag=False, beta=1.5, alpha=0.1):
    retrieved_docs = ""
    if e5_flag:
        query_embedding = EMBEDDER.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    else:
        query_embedding = EMBEDDER.encode(question, convert_to_tensor=True)
    scores = cos_sim(query_embedding, db["embeddings"])[0].cpu()
    if e5_flag:
        data_tmp = [d[9:] for d in db["docs"]]
        question_tmp = question[7:]
        lcs_score = calc_lcs(question_tmp, data_tmp, beta)
    else:
        lcs_score = calc_lcs(question, db["docs"], beta)

    scores = (scores + alpha * lcs_score)

    top_k_idx = torch.topk(scores, k=k_documents)[1]
    if e5_flag:
        top_k_documents = [db["docs"][idx][9:] for idx in top_k_idx]
    else:
        top_k_documents = [db["docs"][idx] for idx in top_k_idx]
    retrieved_docs = "\n".join(top_k_documents)
    return retrieved_docs
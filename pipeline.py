from retrieve import hybrid_search
from generate import rerank as rerank_chunks, rewrite_query, hyde


def retrieve_top(question, k=5, rewrite=True, use_hyde=False, rerank=True):
    q = question
    if use_hyde:
        q = hyde(question)
    elif rewrite:
        q = rewrite_query(question)
    if rerank:
        results = hybrid_search(q, k=k * 2)
        results = rerank_chunks(q, results, top_n=k)
    else:
        results = hybrid_search(q, k=k)
    return q, results

from fastapi import FastAPI, Depends, Header
from app.schemas import QueryRequest, QueryResponse
from app.utils import verify_token
from app.rag_pipeline import extract_text_from_pdf, chunk_text, embed_chunks, upsert_vectors, retrieve_context, ask_gpt4

app = FastAPI()

@app.post("/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest, authorization: str = Header(...)):
    verify_token(authorization)

    raw_text = extract_text_from_pdf(request.documents)
    chunks = chunk_text(raw_text)
    vectors = embed_chunks(chunks)
    upsert_vectors(vectors)

    answers = []
    for question in request.questions:
        context = retrieve_context(question)
        answer = ask_gpt4(question, context)
        answers.append(answer)

    return {"answers": answers}

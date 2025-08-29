import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings   # ok if you installed langchain-huggingface
# from langchain_community.embeddings import HuggingFaceEmbeddings  # <- use this import if you use langchain-community
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

# ---- LLM setup (Mistral via HF Inference) ----
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_llm(repo_id: str):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,
        top_p=0.95,
        return_full_text=False,                    # cleaner for instruct models
        huggingfacehub_api_token=HF_TOKEN          # <-- pass token here, not in model_kwargs
        # model_kwargs={}                            # usually not needed; keep empty unless required
    )

# ---- Prompt ----
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Don't make anything up.
Answer only from the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk.
"""
def set_custom_prompt(tpl: str):
    return PromptTemplate(template=tpl, input_variables=["context", "question"])

# ---- Vector DB ----
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ---- QA chain ----
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# ---- Run ----
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])

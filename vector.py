from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Popular-Books.csv").head(100)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location) #check if our data exists

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # use f-string to deal with None and Nan automatically
        content = f"{row'Title']} {row['Description']}"

        document = Document(
            page_content=content,
            metadata={
                "title": str(row["Title"]),
                "author": str(row["Author"]),
                "score": float(row["Score"]),
                "rating": int(row["Ratings"]),
                "published": int(row["Published"])if pd.notna(row["Published"]) else 0,
                "image_url": str(row["Image"])
                },
                id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="book_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

     
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
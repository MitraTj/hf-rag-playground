from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

def main():
    # 1. Load documents (adjust filename)
    loader = TextLoader("my_docs.txt")
    docs = loader.load()

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # 3. Create embeddings and vectorstore
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(splits, embedding)

    # 4. Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 5. Initialize HuggingFace LLM endpoint
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token="insert_your_token"
    )

    # 6. Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # 7. Query example
    query = "What is the purpose of RAG systems?"
    response = qa_chain.invoke({"query": query})

    print("\nAnswer:\n", response["result"])
    print("\nSource docs:\n", response["source_documents"])

if __name__ == "__main__":
    main()

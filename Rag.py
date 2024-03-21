from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# loading documents from directory and splitting them
def doc_loader(path):
    directory_path = path
    directory_loader = DirectoryLoader(directory_path, use_multithreading=False)
    text_docs = directory_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    docs = text_splitter.split_documents(text_docs)
    return docs

doc_loader('sample_dataset')


model_path = 'sentence-transformers/all-MiniLM-l6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name = model_path,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)
    

db = FAISS.from_documents(doc_loader('sample_dataset'), embeddings)

retriever = db.as_retriever(search_kwargs = {'k': 6})






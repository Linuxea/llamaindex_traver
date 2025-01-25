from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import os


Settings.llm = OpenRouter(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

# Create embeddings
# text_type=`document` to build index
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
)


from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("./data").load_data()


# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db1")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)


query_engine = index.as_query_engine()

print(query_engine.query("你了解我吗？"))
print(query_engine.query("我几岁？"))

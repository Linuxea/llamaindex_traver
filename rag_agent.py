from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

Settings.llm = OpenRouter(
    model="openai/gpt-4o-2024-11-20",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    max_tokens=120000,
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


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)


from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("./data").load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db2")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

from llama_index.core.tools import QueryEngineTool

# Wrap query engine as a tool
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine, name="my_info_retriever", description="my info detail"
)


agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, rag_tool], llm=Settings.llm, verbose=True
)

while True:
    question = input("you:")
    if question == "quit":
        break
    response = agent.chat(question)
    print(response)

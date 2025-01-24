import os
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings
import time

Settings.llm = OpenRouter(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)


from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("./data").load_data()


from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

start_time = time.time()
# Create embeddings
# text_type=`document` to build index
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
)
print(f"Model loading took {time.time() - start_time:.2f} seconds")

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine()

print(query_engine.query("你了解我吗？"))
print(query_engine.query("我几岁？"))

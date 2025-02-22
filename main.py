from os import environ
from dotenv import load_dotenv
from datasets import load_dataset
from llama_index.core import Document
#from llama_index import VectorStoreIndex
#from llama_index import StorageContext

load_dotenv()

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

if (OPENAI_API_KEY):
    print("success")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# Create an llm object to use for the QueryEngine and the ReActAgent
llm = OpenAI(model="gpt-4")

import phoenix as px
session = px.launch_app()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

try:
    storageContext = StorageContext.from_defaults(persist_dir= './sources/docs')

    contextIndex = load_index_from_storage(storageContext)

    storageContext = StorageContext.from_defaults(persist_dir= '.sources/statues')

    statutesIndex = load_index_from_storage(storageContext)

    index_loaded = True
    
except:
        index_loaded = False

# Step 1: Load Pile of Law Dataset

dataset = {
     "UScode": load_dataset("pile-of-law/pile-of-law", name= "uscode", split='train[:500]'),
     "AContracts": load_dataset("pile-of-law/pile-of-law", name="atticus_contracts", split='train[:500]'),
     "courtOpinions": load_dataset("pile-of-law/pile-of-law", name="courlistener_opinions", split='train[:500]')
}


#Step 2. Preprocess the data (tokenization and removing any noise that is not needed)
def preprocessData():
     pass



'''if not index_loaded:
    # Load a subset of documents for testing
    legal_docs = [Document(text=doc['text']) for doc in dataset['train'].select(range(100))]

    # Step 2: Build the Index
    legal_index = VectorStoreIndex.from_documents(legal_docs, show_progress=True)

    # Step 3: Persist (Save) the Index
    legal_index.storage_context.persist(persist_dir="./storage/legal_docs")
    '''



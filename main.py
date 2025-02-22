from os import environ
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

if (OPENAI_API_KEY):
    print("SUCCESS")
else:
    print("failure")

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

#set up PHEONIX
import phoenix as px
session = px.launch_app()
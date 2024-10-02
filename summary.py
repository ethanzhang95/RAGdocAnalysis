# summarize a document

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.llms.openai import OpenAI
from llama_index.extractors.marvin import MarvinMetadataExtractor
#from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import StorageContext
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    #EntityExtractor,
    KeywordExtractor,
    BaseExtractor,
)

# load documents in a directory
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
#from IPython.display import Markdown, display
# include query.py file
# from query import *

# how to delete a document from indexing

import os
import openai

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)


query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)



print(query_engine.query("Context: the given document is a finance report for a company \
                          Objective: generate a finance summary from the report, especially about the EBIDA \
                         Response: Return your response which covers the key points of the text and does not miss anything important.\
                         "))

# knowledgebase Graph DB
#graph_store = SimpleGraphStore()
#storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
#index = KnowledgeGraphIndex.from_documents(
#    documents,
#    max_triplets_per_chunk=2,
#    storage_context=storage_context,
#)
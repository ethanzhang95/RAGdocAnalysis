# an advanced RAG pipeline for document/data unstructured ingestion using LlamaIndex
# Aug, 2024 Ethan Zhang 


import nest_asyncio

nest_asyncio.apply()

import multiprocessing as mp 


from llama_index.core import VectorStoreIndex,KnowledgeGraphIndex
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
from llama_index.core.node_parser import UnstructuredElementNodeParser
from pathlib import Path
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate,Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore
# include query.py file
# from query import *

# how to delete a document from indexing

import os
import openai


mp.set_start_method('fork')


def build_pipeline_ingest(documents, strategy):
# semantic chunking algo and meta extractors setup 

    
    try:

        embed_model = OpenAIEmbedding()
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)
        
        if strategy == "hierachical":
            # combine all documents into one
            documents = [
                Document(text="\n\n".join(
                        document.get_content(metadata_mode=MetadataMode.ALL)
                        for document in documents
                    )
                )
            ]

            # Hierachy text split chunks (METHOD 1)
            #text_splitter_ids = ["1024", "510"]
            #text_splitter_map = {}
            #for ids in text_splitter_ids:
            #    text_splitter_map[ids] = TokenTextSplitter(
            #    chunk_size=int(ids),
            #    chunk_overlap=200
            #    )
            #node_parser = HierarchicalNodeParser.from_defaults(node_parser_ids=text_splitter_ids, node_parser_map=text_splitter_map)

            # Hierachy text split chunk into 3 levels METHOD 2
            # majority means 2/3 are retrieved before using the parent
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

        elif strategy == "unstructured":
             # init NodeParser
            node_parser = UnstructuredElementNodeParser()

            # in case you want to re-use it later.
            raw_nodes = node_parser.get_nodes_from_documents(documents)
            # base nodes and node mapping
            base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(
                raw_nodes
            )

        elif strategy == "semantic":
            # semantic chunking 
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
            )

        # construct text splitter to split texts into chunks for processing
        # this takes a while to process, you can increase processing time by using larger chunk_size
        # file size is a factor too of course

        elif strategy == "KGI":
            Settings.llm = llm
            Settings.chunk_size = 512
            graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=graph_store)

            # NOTE: can take a while!
            index = KnowledgeGraphIndex.from_documents(
                documents,
                max_triplets_per_chunk=2,
                storage_context=storage_context,
                include_embeddings=True,
            )
            return index
          
        else:
            node_parser = TokenTextSplitter(
                separator=" ", chunk_size=2048, chunk_overlap=512
            )


        # meta data extractors creation with capability for customed meta
        # in cyber use case, the meta may need to contain policy control #, document issuing standard 
        class CustomExtractor(BaseExtractor):
            def extract(self, nodes):
                metadata_list = [
                    {
                        "custom": (
                            node.metadata["document_title"]
                            + "\n"
                            + node.metadata["excerpt_keywords"]
                        )
                    }
                    for node in nodes
                ]
                return metadata_list



        extractors = [
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
            #EntityExtractor(prediction_threshold=0.5),
            SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
            KeywordExtractor(keywords=10, llm=llm),
            #CustomExtractor()
        ]

        # run the ingestion pipeline with the enabled meta extractors
        pipeline = IngestionPipeline(
            transformations=[node_parser] + extractors
        )
        

        return pipeline.run(
            documents=documents,
            num_workers=0,  # multiprocess does not run on mac local, experiment when in aws containers
            in_place=True,
            show_progress=True,
            )
    
    except Exception as e:
        print(e)
        return None
    


import subprocess

def convert_pdf_to_html(pdf_path):
    command = f"pdf2htmlEX {pdf_path}"
    subprocess.call(command, shell=True)

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.data_structs import Node

from llama_index.core import Document
from typing import List

def unstructured_elements_to_llamaindex_documents(unstructured_elements: List[dict]) -> List[Document]:
    """
    Convert the "elements" output from the Unstructured.io parser to LlamaIndex Documents.
    
    Args:
        unstructured_elements (list[dict]): The "elements" output from the Unstructured.io parser.
    
    Returns:
        list[Document]: A list of LlamaIndex Documents.
    """
    documents = []
    
    for element in unstructured_elements:
        # Extract the text from the element
        element_text = element['text']
        
        # Create a LlamaIndex Document object
        document = Document(
            text=element_text,
            doc_id=f"element_{element['page_num']}_{element['type']}",
            extra_info={
                'page_num': element['page_num'],
                'element_type': element['type'],
                'bounding_box': element['bounding_box']
            }
        )
        
        # Add the document to the list
        documents.append(document)
    
    return documents

def unstructured_elements_to_llamaindex_nodes(unstructured_elements):
    """
    Convert the "elements" output from the Unstructured.io parser to LlamaIndex nodes.
    
    Args:
        unstructured_elements (list[dict]): The "elements" output from the Unstructured.io parser.
    
    Returns:
        list[Node]: A list of LlamaIndex nodes.
    """
    # Create a SimpleNodeParser
    parser = SimpleNodeParser()
    
    # Initialize an empty list to store the nodes
    nodes = []
    
    # Loop through the elements in the Unstructured.io output
    for element in unstructured_elements:
        # Extract the text from the element
        element_text = element['text']
        
        # Create a Node object from the element text
        node = Node(
            text=element_text,
            doc_id=element['page_num'],
            extra_info={
                'page_num': element['page_num'],
                'element_type': element['type'],
                'bounding_box': element['bounding_box']
            }
        )
        
        # Add the node to the list
        nodes.append(node)
    
    return nodes

#write a python function to parse PDF file using unstructured.IO
from unstructured.partition.auto import partition

def parse_pdf_with_unstructured(pdf_path):
    """
    Parse a PDF file using the Unstructured.io library.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        dict: The output from the Unstructured.io parser.
    """
    try:
        # Parse the PDF file using Unstructured.io
        unstructured_output = partition(pdf_path)   #this depends on poppler installation, potentially pdf2text pdf2image
        
        return unstructured_output  #elements
    
    except Exception as e:
        print(f"Error parsing PDF file: {e}")
        return None
    

# main doc loading and embedding logic
from pprint import pprint

strategy = "KGI" # "hierachical"|"semantic"|"text"|"unstructured"|"unstructured_lib" 

if strategy == "unstructured_lib":
    input_doc = "./data/invoice.pdf"
    elements = parse_pdf_with_unstructured(input_doc)

    print(elements)
    documents = unstructured_elements_to_llamaindex_documents(elements)

    # read the data
    #print("loading test.htm============")
    #reader = FlatReader()
    #documents = reader.load_data(Path('./data/magzine.html'))

elif strategy == "unstructured":
    #input_pdf = "The_Worlds_Billionaires.pdf"
    #convert_pdf_to_html(input_pdf)
    # read the data
    
    # using HTML as raw to start, e.g. converting pdf or office to HTML first
    input_doc = "./data/magzine.html"
    print("loading testing html============", input_doc)
    reader = FlatReader()
    documents = reader.load_data(Path(input_doc))

else: 
    print("loading data folder=======================")
    documents = SimpleDirectoryReader("./data").load_data()


   
nodes = build_pipeline_ingest(documents,strategy)

if nodes and strategy !="KGI":
    for i in range(1):
        pprint(nodes[i].metadata)


if strategy == "hierachical" and nodes :
    leaf_nodes = get_leaf_nodes(nodes)
    if leaf_nodes:
        print("Leaf_nodes=========")    
        for i in range(3):
            pprint(leaf_nodes[i].metadata)

    # parent nodes are added to simple docstore, while leaf_nodes are added in index
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    index_nodes = leaf_nodes
    index = VectorStoreIndex(index_nodes, storage_context=storage_context)
elif strategy == "KGI":
    index = nodes
else:
    index_nodes = nodes
    index = VectorStoreIndex(index_nodes)


index.set_index_id(strategy)
index.storage_context.persist(persist_dir="./store"+strategy)


# if to use mongoatlas as the vector store
#mongo_uri = ("mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority")
#mongodb_client = pymongo.MongoClient(MONGO_URI)
#store = MongoDBAtlasVectorSearch(mongodb_client, MONGO_DB, MONGO_COLLECTION, MONGO_INDEX)
#storage_context = StorageContext.from_defaults(vector_store=store)
#index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context)

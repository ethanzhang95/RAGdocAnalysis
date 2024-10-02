# Aug, 2024 Ethan Zhang 
# the retrieval and query pipeline for RAG to analyze on the document ingested using LLM 

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RetrieverEvaluator 
from llama_index.core import PromptTemplate

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.postprocessor import LLMRerank 
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.retrievers import AutoMergingRetriever



# build index
def build_pipeline_query(storage_context,prompt_str,topic,strategy):

    try: 


        # load index   
        index = load_index_from_storage(storage_context, index_id=strategy)

        # retriever if using hierachy node chunks 
        if strategy=="KGI":
            query_engine = index.as_query_engine(
             include_text=True, response_mode="tree_summarize",embedding_mode="hybrid",
             similarity_top_k=5,
            )

        elif strategy=="hierachical" or strategy=="unstructured":
        
            retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=10),
            storage_context=storage_context,
            simple_ratio_thresh=0.4,
            )

        # configure retriever
        else:
            retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
            )

        # create LLM handle
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)

        if strategy !="KGI":
            # configure response synthesizer
            response_synthesizer = get_response_synthesizer()
            # assemble query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            )

           
            
            # prompt rewriting 
            #reranker = CohereRerank()
            reranker = LLMRerank(choice_batch_size=5, top_n=5) 
            summarizer = TreeSummarize(llm=llm)

            prompt_tmpl = PromptTemplate(prompt_str)

            p = QueryPipeline(verbose=True)
            p.add_modules(
                {
                    "llm": llm,
                    "prompt_tmpl": prompt_tmpl,
                    "retriever": retriever,
                    "summarizer": summarizer,
                    "reranker": reranker,
                }
            )

            # add links for building DAG
            p.add_link("prompt_tmpl", "llm")
            p.add_link("llm", "retriever")
            p.add_link("retriever", "reranker", dest_key="nodes")
            p.add_link("llm", "reranker", dest_key="query_str")
            p.add_link("reranker", "summarizer", dest_key="nodes")
            p.add_link("llm", "summarizer", dest_key="query_str")

            # look at summarizer input keys
            print(summarizer.as_query_component().input_keys)


            # query
            response = p.run(topic=topic)
        elif strategy=="KGI":       
            response = query_engine.query(
               topic,
            )
        # evaluate reponse  
        evaluator = FaithfulnessEvaluator(llm=llm)
        eval_result = evaluator.evaluate_response(response=response)

    
        return response, eval_result.passing
    
    except Exception as e:
        print(e)
        return None
    

from llama_index.core import StorageContext, load_index_from_storage

prompt_str = "{topic}"


strategy="KGI"
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./store"+strategy)


#

response, passing = build_pipeline_query(storage_context, prompt_str, 
                        "Context: the given document is a finance report for a company \
                         Objective: extract the finance numbers and analyze the EBITA \
                                                     Tone: affirmative \
                        Response: Return your response with the complete list of findings and do not miss anything important.\
                         ", strategy)

print(response)
print(str(passing))



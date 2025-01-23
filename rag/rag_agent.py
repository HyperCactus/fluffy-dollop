import dspy
from retrieval.retrieval_model import DocumentRetriever, SummarizeDocument, local_embed
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os
import mlflow

def search_web(query: str):
    ddgs = DDGS()
    results = ddgs.text(query, max_results=8)
    return results

class DeepSearch(dspy.Signature):
    """Use avalable information to answer questions relyably.
    """
    query = dspy.InputField()
    answer = dspy.OutputField(disc="The final summarized report to the user in markdown format.")

class RAG(dspy.Module):
    def __init__(self, persist_dir: os.PathLike = "retrieval/chroma_data"):
        self.retriever = DocumentRetriever(persist_dir)
        self.summarizer = dspy.Predict(SummarizeDocument)
        self.rag = dspy.ReAct(
            DeepSearch, 
            max_iters=4,
            tools=[
                search_web,
                self.retriever.add_from_url,
                self.retriever
                ]
        )
    
    def forward(self, query):
        results = self.rag(query=query)
        return results

if __name__ == "__main__":
    load_dotenv()
    # lm = dspy.LM("deepinfra/deepseek-ai/DeepSeek-R1")
    lm = dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct")
    dspy.configure(lm=lm)

    mlflow.dspy.autolog()
    mlflow.set_experiment("RAG Experiment")

    question = "What are the benifits of lead iron battery compared to lithium ion battery?"
    # print(lm(question))
    rag = RAG()
    response = rag(question)
    answer = response.answer
    print(f"Question: {question}")
    print(f"Answer:\n{answer}")
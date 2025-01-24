import dspy
from retrieval.retrieval_model import DocumentRetriever, SummarizeDocument, local_embed
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os
import mlflow
# TODO: Add ArxivRetriever and WikipediaRetriever from langchain!

class DeepSearch(dspy.Signature):
    """Use avalable information to answer questions relyably.
    """
    question = dspy.InputField()
    response = dspy.OutputField(disc="The final summarized report to the user in markdown format.")

class RAG(dspy.Module):
    def __init__(self, persist_dir: os.PathLike = "retrieval/chroma_data"):
        self.retriever = DocumentRetriever(persist_dir)
        self.summarizer = dspy.Predict(SummarizeDocument)
        self.rag = dspy.ReAct(
            DeepSearch, 
            max_iters=3,
            tools=[
                self.search_web,
                self.search_site
                ]
        )
    
    def search_web(self, question: str):
        """Basic web search to get top pages.
        """
        ddgs = DDGS()
        results = ddgs.text(question, max_results=8)
        return results
    
    def search_site(self, query: str, url: str):
        """Search the contents of a specific site for relevant context.
        Args:
            query (str): The query to search for.
            url (str): The URL of the site to search.
        """
        self.retriever.add_from_url(url)
        return self.retriever.search_top_chunks(query, where={"source": url})

    def forward(self, question):
        results = self.rag(question=question)
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
    answer = response.response
    print(f"Question: {question}")
    print(f"Answer:\n{answer}")
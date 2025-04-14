from retrieval.retrieval_model import DocumentRetriever
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import dspy
from dotenv import load_dotenv

DSPY_DESC = """
DSPy is the framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional Python code and use DSPy to teach your LM to deliver high-quality outputs. This lecture is a good conceptual introduction. Meet the community, seek help, or start contributing via our GitHub repo and Discord server.
""".strip()

MANIM_DESC = """
The community edition of Manim has been forked from 3b1b/manim, a tool originally created and open-sourced by Grant Sanderson, also creator of the 3Blue1Brown educational math videos. While Grant Sanderson’s repository continues to be maintained separately by him, he is not among the maintainers of the community edition. We recommend this version for its continued development, improved features, enhanced documentation, and more active community-driven maintenance. If you would like to study how Grant makes his videos, head over to his repository (3b1b/manim).
""".strip()

class GenerateQueries(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    queries: list = dspy.OutputField(desc="List of queries to search the documentation", optional=True)
    done: bool = dspy.OutputField(
        desc="Whether the context available completely answers the question", default=False
    )

class RAG(dspy.Module):
    def __init__(self, max_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.max_iters = max_iters
        self.gen_queries = dspy.ChainOfThought(GenerateQueries)
        self.retriever = DocumentRetriever(
            "retrieval/chroma_data",
            collection_name="documentations",
            chunk_size=1000,
            chunk_overlap=100,
            embedding_function=OpenAIEmbeddingFunction(
                model_name="text-embedding-3-large",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            summerizer_lm=dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct")
        )
        self.answer = dspy.ChainOfThought("question, context -> response")
    
    def add_docs(self, base_url, name, description, depth=4):
        self.retriever.add_from_base_url(base_url, name=name, description=description, depth=depth)
    
    def forward(self, question):
        done = False
        context = ""
        i = 0
        while not done and i < self.max_iters:
            i += 1
            queries = self.gen_queries(question=question, context=context).queries
            for query in queries:
                context = self.retriever.search_top_chunks(query)
                if context:
                    break
            done = self.gen_queries(question=question, context=context).done
        
        response = self.answer(question=question, context=context).response
        return dspy.Prediction(question=question, response=response)

if __name__ == "__main__":
    load_dotenv()
    
    # retriever.add_from_base_url("https://dspy.ai/", name="dspy-docs", description=DSPY_DESC, depth=4)
    lm = dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct")
    dspy.configure(lm=lm)
    rag = RAG()

    # rag.add_docs("https://dspy.ai/", "dspy-docs", DSPY_DESC, depth=4)
    rag.add_docs("https://docs.manim.community/en/stable/", "manim-docs", MANIM_DESC, depth=4)

    # q = input("Ask DSPy Docs: ")
    # print(rag(q).response)

import dspy
from dspy import Retrieve, Program, LM, Signature
import openai
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Metadata, EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os
from os import PathLike
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, Dict, Union
import logging
from datetime import datetime
import pymupdf
import docx
from openpyxl import load_workbook
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def local_embed(texts: List[str]) -> List[List[float]]:
    client = openai.OpenAI(
        base_url="http://localhost:1234/v1"
    )
    response = client.embeddings.create(
        input=texts,
        model="nomic-ai/nomic-embed-text-v1.5-GGUF"
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return embeddings

class SummarizeDocument(Signature):
    """Summarize a document."""
    context: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="No more then one short parograph.")

class DocumentRetriever(Retrieve):
    def __init__(
            self, 
            chroma_persist_dir: str,
            db_name: Optional[str] = None,
            collection_name: Optional[str] = None,
            embedding_function: Optional[EmbeddingFunction] = None,
            chunk_size: Optional[int] = 500,
            chunk_overlap: Optional[int] = 50,
            summerizer: Optional[Program] = None,
            summerizer_lm: Optional[LM] = None,
            k: int = 5
        ):
        load_dotenv()
        super().__init__(k=k)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_function = embedding_function or OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.summarizer = summerizer or dspy.Predict(SummarizeDocument)
        if summerizer_lm:
            self.summarizer.set_lm(summerizer_lm)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            database=db_name or "retrieval_db",
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name or "documents", 
            embedding_function=self.embedding_function,
            metadata={
                "time_created": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "hnsw:search_ef": 500,
                "hnsw:construction_ef": 1000,
                "hnsw:M": 128
            }
        )
        
        logger.info("Initialized DocumentRetriever with ChromaDB collection.")
    
    def add_to_collection(self, ids: list, content: list[str], metadatas: list[dict]) -> None:
        """Add documents and their metadata to the ChromaDB collection."""
        self.collection.upsert(
            ids=ids,
            documents=content,
            metadatas=metadatas
        )
    
    def chunk_and_add(self, content: str, doc_name: str) -> None:
        """Chunk a document and add it to the ChromaDB collection."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        text_chunks = splitter.split_text(content)
        # Create metadata for chunks, associating them with the main document
        chunk_metadatas = [
            {"parent_doc": doc_name, 
             "chunk_index": i, 
             "type": "chunk"}
            for i in range(len(text_chunks))
        ]

        # Add chunks to the collection
        chunk_ids = [f"{doc_name}_chunk_{i}" for i in range(len(text_chunks))]
        self.add_to_collection(chunk_ids, text_chunks, chunk_metadatas)
        # Log completion
        logger.info(f"Document '{doc_name}' added with {len(text_chunks)} chunks.")

    def add_doc(
            self, name: str, description: str, content: str, doc_metadata: dict={}
            ) -> None:
        """Add a document and its chunks to the ChromaDB collection."""
        doc_metadata["type"] = "document"
        self.add_to_collection([name], [description], [doc_metadata])

        self.chunk_and_add(content, name)

    def add_from_str(
            self, name: str, content: str, description=None, doc_metadata: dict={}
            ) -> None:
        """Create and add a document from a string."""
        if not "source" in doc_metadata:
            doc_metadata["source"] = "string"
        
        if description is None:
            summerizer_response = self.summarizer(context=content)
            description = summerizer_response.summary
        else:
            description = description
        
        self.add_doc(name=name, description=description, 
                     content=content, doc_metadata=doc_metadata)

    def add_from_txt(self, path: PathLike, description=None, doc_metadata: dict={}) -> None:
        """Create and add a document from a .txt file."""
        doc_metadata["source"] = path
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        self.add_from_str(name, description, content)
    
    def add_from_pdf(self, path: PathLike, description=None, doc_metadata: dict={}) -> None:
        """Create and add a document from a .pdf file."""
        doc_metadata["source"] = path
        name = os.path.basename(path)
        text = ""
        with pymupdf.open(path) as doc:
            for page in doc:
                text += page.get_text()
        self.add_from_str(name, text, description=description)
    
    def add_from_docx(self, path: PathLike, description: str=None, doc_metadata: dict={}) -> None:
        """Create and add a document from a .docx file."""
        doc_metadata["source"] = path
        name = os.path.basename(path)
        text = ""
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text
        self.add_from_str(name, text, description=description, doc_metadata=doc_metadata)
    
    def add_from_xlsx(self, path: PathLike, description: str=None, doc_metadata: dict={}) -> None:
        """Create and add a document from a .xlsx file."""
        doc_metadata["source"] = path
        name = os.path.basename(path)
        text = ""
        wb = load_workbook(path)
        for sheet in wb.sheetnames:
            worksheet = wb[sheet]
            for row in worksheet.iter_rows(values_only=True):
                text += "\t".join([str(cell) for cell in row if cell is not None]) + "\n"
        self.add_from_str(name, text, description=description, doc_metadata=doc_metadata)
    
    def _url_to_text(self, url: str) -> str:
        """Fetch the text content of a webpage."""
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch URL: {e}")
        
        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main content of the page
        # This is a heuristic for cleaner content; adjust as needed.
        main_content = soup.find('main') or soup.body
        if main_content:
            text = ' '.join(main_content.stripped_strings)
        else:
            raise ValueError("Failed to find main content in the webpage.")
        return text

    def add_from_url(self, url: str, description: str = None, doc_metadata: dict={}) -> None:
        """Create and add a document from a URL."""
        doc_metadata["source"] = url
        name = url.split("/")[-1] or "webpage"
        text = self._url_to_text(url)
        # Add the document
        self.add_from_str(name, text, description=description, doc_metadata=doc_metadata)
    
    def add_pages_recursive(
            self, 
            base_url: str, 
            depth: int, 
            name: str = None
        ) -> None:
        """Add a document to the DB consisting of the content of a webpage and its subpages."""
        name = name or base_url.split("/")[-1] or "webpage"
        text = self._url_to_text(base_url)
        # Add the main page content as chunks
        self.chunk_and_add(text, name)
        # Recursively add subpages
        if depth > 0:
            soup = BeautifulSoup(requests.get(base_url).text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                absolute_url = urljoin(base_url, href)
                # get the url as a string
                str_url = str(absolute_url)
                if "#" in str_url:
                    continue
                subpage_url = urlparse(absolute_url).path.lstrip(urlparse(base_url).path)
                if urlparse(absolute_url).netloc == urlparse(base_url).netloc \
                    and not subpage_url.startswith(('#', '_')):
                    print(f'\nLINK: {absolute_url}')
                    self.add_pages_recursive(
                        absolute_url, 
                        depth-1, 
                        name=name
                    )
    
    def add_from_base_url(
            self, base_url: str, 
            depth: int = 20, 
            name: str = None,
            description: str = None, 
            doc_metadata: dict={}
        ) -> None:
        """Add a document to the DB consisting of the content of a webpage and its subpages."""
        doc_metadata["source"] = base_url
        name = name or base_url.split("/")[-1] or "webpage"
        text = self._url_to_text(base_url)
        description = description or self.summarizer(context=text).summary
        self.add_to_collection([name], [description], [doc_metadata]) # Add base page as document
        self.add_pages_recursive(base_url, depth, name=name) # Add chunks from page and subpages

    def search_top_chunks(self, query: Union[str, list[str]], k: int = 5) -> List[Tuple[str, str]]:
        """Search for the top k chunks across all documents."""
        results = self.collection.query(
            query_texts=query,
            n_results=k,
            where={"type": "chunk"},  # Target chunks specifically
            include=["documents", "metadatas"]
        )
        logger.debug(f"Top chunks results: {results}")

        # Extract relevant details
        top_chunks = []
        for chunk_doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            top_chunks.append(chunk_doc)
        return top_chunks

    def get_top_docs(self, query: Union[str, list[str]], k: int = 1) -> List[str]:
        """Returns a list of k document names in descending order of relevance."""
        results = self.collection.query(
            query_texts=query,
            n_results=k,
            where={"type": "document"}
        )
        return results["ids"][0]

    def top_from_doc(self, query: Union[str, list[str]], doc_name: str, k: int = 5) -> List[str]:
        """Get the top k chunks from a specific document."""
        results = self.collection.query(
            query_texts=query,
            n_results=k,
            where={
                "$and": [
                    {"type": "chunk"}, 
                    {"parent_doc": {"$eq": doc_name}}
                ]
            }
        )
        logger.debug(f"Top from doc results: {results}")
        return results['documents'][0]
    
    def forward(
            self,
            query: str,
            k: Optional[int] = None,
            **kwargs,
        ) -> List[str]:
        k = k or self.k
        return self.search_top_chunks(query, k)

    def delete_collection(self) -> None:
        self.chroma_client.delete_collection("documents")
        logger.info("Collection deleted.")
    
    def delete_document(self, doc_name: str) -> None:
        self.collection.delete(ids=[doc_name]) # Delete the document description
        # Delete the associated chunks
        self.collection.delete(where={"parent_doc": {"$eq": doc_name}})
        logger.info(f"Document '{doc_name}' deleted.")

if __name__ == "__main__":
    lm = dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct")
    dspy.configure(lm=lm)
    # Initialize the DocumentRetriever
    embedding_func = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    retriever = DocumentRetriever(
        chroma_persist_dir="retrieval/chroma_data", embedding_function=embedding_func
    )
    
    # Add documents to the collection
    # retriever.add_from_pdf("retrieval/test_data/2407.09450v1_episodic_memory.pdf")
    # retriever.add_from_base_url("https://dspy.ai/", name="dspy-docs", depth=1)
    # retriever.add_from_url("https://en.wikipedia.org/wiki/Fall_of_Babylon")
    
    # Search for relevant chunks
    query = "how does BootstrapFinetune work?"
    top_chunks = retriever(query, k=5)
    r = "\nTop chunks for query:\n"
    for i in range(len(top_chunks)):
        r += f"CHUNK {i+1}:\n{top_chunks[i]}\n\n"
    print(r)
    
    # retriever.delete_collection()
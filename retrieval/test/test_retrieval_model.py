import unittest
from retrieval.retrieval_model import DocumentRetriever, local_embed
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
import logging
from dotenv import load_dotenv
import dspy
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestDocumentRetrieverIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logger.info("[SETUP] Setting up test environment.")
        
        # Set up ChromaDB client with a temporary persist directory
        lm = dspy.LM("deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct") # Hosted LM
        # lm = dspy.LM(
        #     model='llama-3.2-3b-instruct', 
        #     provider='lm_studio',
        #     base_url="http://localhost:1234/v1", 
        #     api_key="lm-studio"
        # ) # Local LM
        dspy.configure(lm=lm)

        cls.chroma_persist_dir = "./chroma_data_test"
        cls.chroma_client = chromadb.Client(
            Settings(
                persist_directory=cls.chroma_persist_dir,
                anonymized_telemetry=False,
                allow_reset=True  # Allow resetting the client
            )
        )
        cls.collection = cls.chroma_client.get_or_create_collection("documents")
        
        # Create a custom embedder
        # cls.embedding_func = OllamaEmbeddingFunction( # Local embedder
        #     url="http://localhost:1234/v1/embeddings",
        #     model_name="text-embedding-nomic-embed-text-v1.5@q4_k_m"
        # )
        cls.embedding_func = OpenAIEmbeddingFunction( # OpenAI embedder
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Pass the existing chroma_client and custom embedder to DocumentRetriever
        cls.retriever = DocumentRetriever(
            chroma_persist_dir=cls.chroma_persist_dir,
            embedding_function=cls.embedding_func,
            chroma_client=cls.chroma_client
        )

        # Add test documents
        cls.add_test_docs()

    @classmethod
    def add_test_docs(cls):
        logger.info("[SETUP] Adding test documents to the retriever.")
        
        cls.retriever.add_from_str(
            name="Doc1",
            description="This document is about machine learning techniques.",
            content="Machine learning is a field of AI. It includes supervised learning and unsupervised learning."
        )
        cls.retriever.add_from_str(
            name="Doc2",
            content="Mars Mars Mars Space exploration includes missions to Mars. It also involves studying the Moon."
        )
        cls.retriever.add_from_str(
            name="Doc3",
            description="This document is about programming languages.",
            content="Programming languages like Python, Java, and C++ are widely used in software development."
        )

        # Verify documents and chunks are added
        stored_data = cls.collection.peek()
        logger.debug(f"[SETUP] Stored data after adding documents: {stored_data}")

    def test_search_top_chunks(self):
        logger.info("[TEST] Testing search_top_chunks.")
        
        query = "Tell me about AI."
        results = self.retriever.search_top_chunks(query, k=3)
        
        # Debug results
        logger.debug(f"[TEST] Top chunks results: {results}")
        
        self.assertEqual(len(results), 3)
        doc_names = [name for name, _ in results]
        self.assertIn("Doc1", doc_names)
        for _, chunk in results:
            self.assertIsNotNone(chunk, "[TEST] Chunk content is None.")

    def test_get_top_docs(self):
        logger.info("[TEST] Testing get_top_docs.")
        
        query = "Tell me about AI."
        top_docs = self.retriever.get_top_docs(query, k=1)
        
        # Debug results
        logger.debug(f"[TEST] Top docs results: {top_docs}")
        
        self.assertEqual(len(top_docs), 1)
        self.assertEqual(top_docs[0], "Doc1")

    def test_top_from_doc(self):
        logger.info("[TEST] Testing top_from_doc.")
        
        query = "Tell me about Mars."
        doc_name = "Doc2"
        chunks = self.retriever.top_from_doc(query, doc_name, k=2)
        
        # Debug results
        logger.debug(f"[TEST] Top chunks from Doc2: {chunks}")
        
        self.assertIn(len(chunks), [2, 1])
        for chunk in chunks:
            self.assertIsNotNone(chunk, "[TEST] Retrieved chunk is None.")
        self.assertIn("Mars", chunks[0], "[TEST] 'Mars' not found in top chunk.")

    def test_collection_integrity(self):
        logger.info("[TEST] Verifying collection integrity.")
        
        stored_data = self.collection.peek()
        logger.debug(f"[TEST] Current collection data: {stored_data}")
        
        self.assertTrue(len(stored_data["metadatas"]) > 0, "[TEST] Metadata is missing from collection.")
        self.assertTrue(len(stored_data["embeddings"]) > 0, "[TEST] Embeddings are missing from collection.")
        for metadata in stored_data["metadatas"]:
            self.assertIn("type", metadata, "[TEST] 'type' missing in metadata.")
            if metadata["type"] == "chunk":
                self.assertIn("parent_doc", metadata, "[TEST] 'parent_doc' missing in metadata.")
            elif metadata["type"] == "document":
                self.assertIn("source", metadata, "[TEST] 'description' missing in metadata.")
            

    @classmethod
    def tearDownClass(cls):
        logger.info("[TEARDOWN] Cleaning up test environment.")
        
        # Clean up test data
        cls.collection.delete(where={"name": {"$in": ["Doc1", "Doc2", "Doc3"]}})
        
        # Ensure changes are persisted
        cls.chroma_client.reset()
        logger.info("[TEARDOWN] Test environment reset successfully.")

if __name__ == '__main__':
    unittest.main()

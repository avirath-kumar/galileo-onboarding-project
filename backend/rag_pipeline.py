import os
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import hashlib
import json

load_dotenv()

class SimpleRAGPipeline:

    # Initialize RAG pipeline w. pdf directory, collection name, and directory for chromadb
    def __init__(self, pdf_directory: str = "static", collection_name: str = "product_docs", persist_directory: str = "./chroma_db"):
        self.pdf_directory = Path(pdf_directory)
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize chromadb client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            print(f"Created new collection: {collection_name}")
        except:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Track processed files
        self.processed_files_path = Path(persist_directory) / "processed_files.json"
        self.processed_files = self._load_processed_files()
    
    # Load the record of processed files
    def _load_processed_files(self) -> Dict[str, str]:
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}
    
    # Save the record of processd files
    def _save_processed_files(self):
        self.processed_files_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f)

    # Get hash of file for change detection
    def _get_file_hash(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    # Load PDFs from the directory and process them into the vector store. 
    # force_reload is a bool that reprocesses all files regardless of cache. Returns # of docs processed
    def load_and_process_pdfs(self, force_reload: bool = False) -> int:
        if not self.pdf_directory.exists():
            print(f"Creating directory: {self.pdf_directory}")
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            return 0
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return 0
        
        total_chunks = 0

        for pdf_path in pdf_files:
            file_hash = self._get_file_hash(pdf_path)

            # Skip if file hasn't changed and not forcing reload
            if not force_reload and str(pdf_path) in self.processed_files:
                if self.processed_files[str(pdf_path)] == file_hash:
                    print(f"Skipping {pdf_path.name} (already processed)")
                    continue
            
            print(f"Processing {pdf_path.name}...")

            try:
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()

                # split docs into chunks
                chunks = self.text_splitter.split_documents(documents)

                # Prepare data for chromadb
                texts = []
                metadatas = []
                ids = []

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{pdf_path.stem}_{i}"
                    texts.append(chunk.page_content)
                    metadatas.append({
                        "source": pdf_path.name,
                        "page": chunk.metadata.get("page", 0),
                        "chunk_index": i
                    })
                    ids.append(chunk_id)
                
                # Generate embeddings and add to collection
                if texts:
                    embeddings_list = self.embeddings.embed_documents(texts)

                    # remove old chuns from this document if reprocessing
                    if str(pdf_path) in self.processed_files:
                        # Delete old chunks
                        self.collection.delete(
                            where={"source": pdf_path.name}
                        )

                    # Add new chunks
                    self.collection.add(
                        embeddings=embeddings_list,
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )

                    print(f" Added{len(chunks)} chunks from {pdf_path.name}")
                    total_chunks += len(chunks)
                
                # Update processed files record
                self.processed_files[str(pdf_path)] = file_hash
                self._save_processed_files()
            
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {str(e)}")
                continue
        
        print(f"Total chunks processed: {total_chunks}")
        return total_chunks
    
    # Search for relevant docs based on query.
    # Args: query = search query, k = no. of results to return. Returns list fo relevant doc chunks w. metadata
    def search(self, query: str, k: int = 3) -> List[Dict]:
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Search in chromadb
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i] # convert distance to similarity
                    })
            
            return formatted_results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    # Get formatted context for query to be used in LLM prompt
    # Args: query is user's question, k is number of relevant chunks to retrieve. Returns formatted context string
    def get_context_for_query(self, query: str, k: int = 3) -> str:
        results = self.search(query, k)
        
        if not results:
            return "No relevant information found in the product documents."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            page = result["metadata"].get("page", "Unknown")
            content = result["content"]

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)

    # Get stats about the vector store
    def get_stats(self) -> Dict:
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "processed_files": list(self.processed_files.keys()),
                "collection_name": self.collection_name
            }
        except:
            return {
                "total_chunks": 0,
                "processed_files": [],
                "collection_name": self.collection_name
            }

# Single instnace for the RAG pipeline
_rag_pipeline: Optional[SimpleRAGPipeline] = None

# Get or create the single RAG pipeline instance
def get_rag_pipeline() -> SimpleRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = SimpleRAGPipeline()
        # Load PDFs on initialization
        _rag_pipeline.load_and_process_pdfs()
    return _rag_pipeline
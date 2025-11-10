import os
import logging
from typing import List, Optional, Tuple
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import save_pickle, load_pickle

# For generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for document Q&A.
    Uses free models: sentence-transformers for embeddings and FLAN-T5 for generation.
    """
    
    def __init__(
        self, 
        embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        gen_model_name: str = 'google/flan-t5-base', 
        device: Optional[str] = None
    ):
        """
        Initialize RAG pipeline with embedding and generation models.
        
        Args:
            embed_model_name: HuggingFace model for embeddings (free)
            gen_model_name: HuggingFace model for text generation (free)
            device: Device to run models on ('cuda' or 'cpu')
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Embedding model (sentence-transformers)
        try:
            self.embedder = SentenceTransformer(embed_model_name)
            logger.info(f"Loaded embedding model: {embed_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Generation model (transformers seq2seq)
        try:
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(self.device)
            self.gen_model.eval()  # Set to evaluation mode
            logger.info(f"Loaded generation model: {gen_model_name}")
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise
        
        # In-memory state (per-document)
        self.index: Optional[faiss.Index] = None
        self.text_chunks: List[str] = []
        self.doc_id: Optional[str] = None
        self.embedding_dim: Optional[int] = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF reading fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            page_count = len(reader.pages)
            
            logger.info(f"Extracting text from {page_count} pages...")
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{page_count} pages")
            
            if not text.strip():
                logger.warning("No text extracted from PDF")
            else:
                logger.info(f"Extracted {len(text)} characters from PDF")
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"PDF extraction failed: {str(e)}")

    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.text_chunks = splitter.split_text(text)
            
            logger.info(f"Created {len(self.text_chunks)} chunks "
                       f"(size={chunk_size}, overlap={chunk_overlap})")
            
            return self.text_chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise

    def build_faiss_index(self, doc_id: str, persist: bool = True) -> faiss.Index:
        """
        Build FAISS index from text chunks and optionally persist to disk.
        
        Args:
            doc_id: Unique document identifier
            persist: Whether to save index and chunks to disk
            
        Returns:
            FAISS index object
            
        Raises:
            ValueError: If no text chunks available
        """
        if not self.text_chunks:
            raise ValueError("No text chunks available. Run chunk_text() first.")
        
        try:
            logger.info(f"Building FAISS index for {len(self.text_chunks)} chunks...")
            
            # Generate embeddings
            embeddings = self.embedder.encode(
                self.text_chunks, 
                show_progress_bar=True,
                batch_size=32,  # Process in batches for efficiency
                convert_to_numpy=True
            )
            embeddings = embeddings.astype('float32')
            
            # Normalize embeddings for better similarity search
            faiss.normalize_L2(embeddings)
            
            dim = embeddings.shape[1]
            self.embedding_dim = dim
            
            # Create FAISS index (using IndexFlatIP for normalized vectors)
            index = faiss.IndexFlatIP(dim)  # Inner Product for normalized vectors
            index.add(embeddings)
            
            self.index = index
            self.doc_id = doc_id
            
            logger.info(f"Built FAISS index with {index.ntotal} vectors (dim={dim})")
            
            # Persist to disk
            if persist:
                self._persist_index(doc_id)
            
            return index
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def _persist_index(self, doc_id: str) -> None:
        """Save FAISS index and chunks to disk."""
        try:
            # Ensure directories exist
            os.makedirs('indexes', exist_ok=True)
            os.makedirs('metadata', exist_ok=True)
            
            # Save index
            index_path = os.path.join('indexes', f'{doc_id}.index')
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save chunks
            chunks_path = os.path.join('metadata', f'{doc_id}_chunks.pkl')
            save_pickle(self.text_chunks, chunks_path)
            logger.info(f"Saved chunks to {chunks_path}")
            
        except Exception as e:
            logger.error(f"Error persisting index: {e}")
            raise

    def load_index(self, doc_id: str) -> faiss.Index:
        """
        Load FAISS index and chunks from disk.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Loaded FAISS index
            
        Raises:
            FileNotFoundError: If index or chunks not found
        """
        index_path = os.path.join('indexes', f'{doc_id}.index')
        chunks_path = os.path.join('metadata', f'{doc_id}_chunks.pkl')
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f'Index not found: {index_path}')
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f'Chunks metadata not found: {chunks_path}')
        
        try:
            # Load index
            index = faiss.read_index(index_path)
            self.index = index
            self.doc_id = doc_id
            
            # Load chunks
            self.text_chunks = load_pickle(chunks_path)
            
            logger.info(f"Loaded index for doc_id={doc_id} "
                       f"({index.ntotal} vectors, {len(self.text_chunks)} chunks)")
            
            return index
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved text chunks
            
        Raises:
            ValueError: If index not loaded
        """
        if self.index is None:
            raise ValueError('FAISS index not loaded. Call load_index() or build_faiss_index().')
        
        try:
            # Encode query
            q_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(q_emb)  # Normalize for cosine similarity
            
            # Search
            distances, indices = self.index.search(q_emb, top_k)
            
            # Collect results
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.text_chunks):
                    results.append(self.text_chunks[idx])
                    logger.debug(f"Retrieved chunk {idx} with score {score:.4f}")
            
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[str], 
        max_source_chars: int = 2000,
        max_length: int = 256, 
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> str:
        """
        Generate answer using retrieved context and query.
        
        Args:
            query: User query
            retrieved_chunks: List of relevant text chunks
            max_source_chars: Maximum context characters to use
            max_length: Maximum tokens in generated answer
            temperature: Sampling temperature (higher = more creative)
            num_beams: Number of beams for beam search
            
        Returns:
            Generated answer
        """
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question."
        
        try:
            # Build context from chunks
            context = "\n\n".join(retrieved_chunks)
            context = context[:max_source_chars]
            
            # Create prompt
            prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Answer the question clearly and concisely using only the information from the context. "
                "If the context doesn't contain enough information, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
            
            # Tokenize
            inputs = self.gen_tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate with improved parameters
            with torch.no_grad():
                outputs = self.gen_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Generated answer ({len(answer)} chars)")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def get_stats(self) -> dict:
        """Get current pipeline statistics."""
        return {
            'doc_id': self.doc_id,
            'chunks_count': len(self.text_chunks),
            'index_loaded': self.index is not None,
            'device': self.device,
            'embedding_dim': self.embedding_dim
        }
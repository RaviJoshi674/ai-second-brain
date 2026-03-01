import os
from typing import List, Dict, Any
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

class DocumentIngestor:
    """Handles parsing and chunking of documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _generate_id(self, text: str) -> str:
        """Generate a stable hash ID for a chunk of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def parse_file(self, file_path: str) -> str:
        """Extract text from supported file types."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = ""
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
            return text
            
        elif ext in ['.txt', '.md', '.csv']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
                return ""
                
        else:
            print(f"Unsupported file type: {ext}")
            return ""

    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Parse file, split into chunks, and format for insertion."""
        if metadata is None:
            metadata = {}
            
        metadata['source'] = os.path.basename(file_path)
        
        text = self.parse_file(file_path)
        if not text:
            return []
            
        chunks = self.text_splitter.split_text(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Create a unique ID for the chunk
            chunk_id = self._generate_id(f"{file_path}_{i}_{chunk}")
            
            # Combine provided metadata with chunk specific info
            chunk_meta = metadata.copy()
            chunk_meta['chunk_id'] = i
            chunk_meta['text'] = chunk
            
            processed_chunks.append({
                "id": chunk_id,
                "text": chunk, # We keep text here internally, but it gets packed into meta for Endee
                "meta": chunk_meta
            })
            
        return processed_chunks

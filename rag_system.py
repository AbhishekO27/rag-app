from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
from typing import List

class RAGSystem:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.chunks = []
        self.embeddings = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += chunk_size - overlap
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def process_pdf(self, pdf_path: str) -> int:
        text = self.extract_text_from_pdf(pdf_path)
        self.chunks = self.chunk_text(text)
        
        self.embeddings = []
        for chunk in self.chunks:
            embedding = self.get_embedding(chunk)
            self.embeddings.append(embedding)
        
        return len(self.chunks)
    
    def find_relevant_chunks(self, question: str, top_k: int = 3) -> List[str]:
        question_embedding = self.get_embedding(question)
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(question_embedding, emb)
            similarities.append((sim, i))
        
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:top_k]]
        
        return [self.chunks[i] for i in top_indices]
    
    def answer_question(self, question: str) -> str:
        relevant_chunks = self.find_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def has_document(self) -> bool:
        return len(self.chunks) > 0
    
    def reset(self):
        self.chunks = []
        self.embeddings = []
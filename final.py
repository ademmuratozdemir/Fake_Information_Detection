import streamlit as st
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Union
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformersEmbedding(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, query: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(query, str):
            return self.model.encode([query], convert_to_tensor=False).tolist()[0]
        elif isinstance(query, list):
            return self.model.encode(query, convert_to_tensor=False).tolist()
        else:
            raise ValueError("Query must be a string or a list of strings.")

class FactChecker:
    def __init__(self, temperature: float = 0.0):
        self.embedding_model = SentenceTransformersEmbedding()
        self.llm = self._initialize_llm(temperature)
        self.vectorstore = None
        self.qa_chain = None

    def _initialize_llm(self, temperature: float) -> OllamaLLM:
        return OllamaLLM(
            model="mistral:7b",
            temperature=temperature,
            top_p=0.9,
            max_tokens=2048,
            repeat_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.2
        )

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentence_ends = r'[.!?](?:\s|\n|$)'
        sentences = re.split(sentence_ends, text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def find_most_similar_sentence(self, query: str, sentences: List[str]) -> tuple:
        """Find the most similar sentence to the query using embeddings."""
        query_embedding = self.embedding_model.embed_query(query)
        sentence_embeddings = self.embedding_model.embed_documents(sentences)
        
        similarities = []
        for sent_emb in sentence_embeddings:
            similarity = np.dot(query_embedding, sent_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(sent_emb)
            )
            similarities.append(similarity)
        
        max_sim_idx = np.argmax(similarities)
        return sentences[max_sim_idx], similarities[max_sim_idx]

    def extract_case_number(self, text: str) -> str:
        """Extract case number from text."""
        import re
        pattern = r'\d{4}/\d+\s*[KEk]\.'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def load_fact_database(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading fact database: {str(e)}")
            raise

    def create_faiss_index(self, fact_db: List[Dict[str, Any]]) -> None:
        try:
            documents = []
            for fact in fact_db:
                doc = Document(page_content=fact['metin'], metadata=fact)
                documents.append(doc)

            texts = [doc.page_content for doc in documents]
            self.vectorstore = FAISS.from_texts(
                texts,
                self.embedding_model,
                metadatas=[doc.metadata for doc in documents]
            )
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise

    def create_retrieval_chain(self) -> None:
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 1,
                    "score_threshold": 0.3
                }
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Error creating retrieval chain: {str(e)}")
            raise

    def verify_statement(self, query: str) -> Dict[str, Any]:
        try:
            case_number = self.extract_case_number(query)
            
            if case_number:
                for doc in self.vectorstore.docstore._dict.values():
                    if case_number in doc.metadata.get('karar_no', ''):
                        sentences = self.split_into_sentences(doc.page_content)
                        most_similar_sentence, similarity_score = self.find_most_similar_sentence(query, sentences)
                        
                        prompt = (
                            "[INSTRUCTION]\n"
                            "Sen bir hukuk uzmanısın ve Yargıtay kararlarını analiz ederek verilen ifadelerin doğruluğunu kontrol ediyorsun.\n\n"
                            "GÖREV:\n"
                            "Aşağıda verilen ifadeyi, karardaki en alakalı cümle ile karşılaştırarak doğruluğunu kontrol et.\n\n"
                            f"VERİLEN İFADE:\n{query}\n\n"
                            f"KARARIN İLGİLİ CÜMLESİ:\n{most_similar_sentence}\n\n"
                            "TALİMATLAR:\n"
                            "1. İfadeyi SADECE verilen cümle ile karşılaştır\n"
                            "2. İfadenin bu cümle ile BİREBİR eşleşmesini kontrol et\n"
                            "3. Dolaylı veya yoruma açık değerlendirmeler yapma\n"
                            "4. En ufak bir tutarsızlık veya eksiklik varsa YANLIŞ olarak değerlendir\n\n"
                            "DEĞERLENDİRME KRİTERLERİ:\n"
                            "- DOĞRU: İfade cümleyle tamamen ve kesin olarak örtüşüyor\n"
                            "- YANLIŞ: İfade cümleyle çelişiyor veya tutarsızlık içeriyor\n"
                            "- BİLGİ YOK: Cümlede ifadeyle ilgili yeterli bilgi yok\n\n"
                            "[OUTPUT]\n"
                            "Yanıtını tam olarak şu formatta ver:\n\n"
                            "1. SONUÇ: [DOĞRU/YANLIŞ/BİLGİ YOK]\n"
                            "2. AÇIKLAMA: [Tek cümlelik kesin açıklama]\n"
                            "3. KANIT: [İlgili cümle]\n"
                            "[/INSTRUCTION]"
                        )
                        
                        response = self.llm.invoke(prompt)
                        return {
                            'response': response,
                            'source_documents': [doc],
                            'relevance_score': similarity_score,
                            'timestamp': datetime.now().isoformat()
                        }
            
            retrieved_docs = self.qa_chain.retriever.invoke(query)
            
            if not retrieved_docs:
                return {
                    'response': "Sorgu ile ilgili karar bulunamadı.",
                    'source_documents': [],
                    'relevance_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                }

            sentences = self.split_into_sentences(retrieved_docs[0].page_content)
            most_similar_sentence, similarity_score = self.find_most_similar_sentence(query, sentences)

            prompt = (
                "[INSTRUCTION]\n"
                "Sen bir hukuk uzmanısın ve Yargıtay kararlarını analiz ederek verilen ifadelerin doğruluğunu kontrol ediyorsun.\n\n"
                "GÖREV:\n"
                "Aşağıda verilen ifadeyi, karardaki en alakalı cümle ile karşılaştırarak doğruluğunu kontrol et.\n\n"
                f"VERİLEN İFADE:\n{query}\n\n"
                f"KARARIN İLGİLİ CÜMLESİ:\n{most_similar_sentence}\n\n"
                "TALİMATLAR:\n"
                "1. İfadeyi SADECE verilen cümle ile karşılaştır\n"
                "2. İfadenin bu cümle ile BİREBİR eşleşmesini kontrol et\n"
                "3. Dolaylı veya yoruma açık değerlendirmeler yapma\n"
                "4. En ufak bir tutarsızlık veya eksiklik varsa YANLIŞ olarak değerlendir\n\n"
                "DEĞERLENDİRME KRİTERLERİ:\n"
                "- DOĞRU: İfade cümleyle tamamen ve kesin olarak örtüşüyor\n"
                "- YANLIŞ: İfade cümleyle çelişiyor veya tutarsızlık içeriyor\n"
                "- BİLGİ YOK: Cümlede ifadeyle ilgili yeterli bilgi yok\n\n"
                "[OUTPUT]\n"
                "Yanıtını tam olarak şu formatta ver:\n\n"
                "1. SONUÇ: [DOĞRU/YANLIŞ/BİLGİ YOK]\n"
                "2. AÇIKLAMA: [Tek cümlelik kesin açıklama]\n"
                "3. KANIT: [İlgili cümle]\n"
                "[/INSTRUCTION]"
            )

            response = self.llm.invoke(prompt)
            return {
                'response': response,
                'source_documents': retrieved_docs,
                'relevance_score': similarity_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during statement verification: {str(e)}")
            raise

def main():
    st.title("Fake Information Detection System")
    st.write("Bu sistem, Yargıtay kararlarına dayalı olarak ifadelerin doğruluğunu kontrol eder.")

    with st.sidebar:
        temperature = st.slider("LLM Sıcaklığı", 0.0, 1.0, 0.0)
        st.info("Bu sistem RAG teknolojisini kullanarak bilgi doğrulama yapar.")

    try:
        fact_checker = FactChecker(temperature=temperature)
        fact_db = fact_checker.load_fact_database("kararlar.json")
        fact_checker.create_faiss_index(fact_db)
        fact_checker.create_retrieval_chain()

        query = st.text_area(
            "İfadeyi Girin:",
            placeholder="Doğrulanmasını istediğiniz ifadeyi yazın...",
            height=100
        )
        
        if st.button("Sorgula", type="primary") and query.strip():
            with st.spinner("İfade analiz ediliyor..."):
                result = fact_checker.verify_statement(query)
                
                st.markdown("### Sonuç")
                st.markdown(result['response'])
                
                if result['source_documents']:
                    st.markdown("### Kaynak Karar")
                    doc = result['source_documents'][0]
                    with st.expander(f"Karar No: {doc.metadata['esas_no']} - {doc.metadata['karar_no']}"):
                        st.write(doc.page_content)
                
    except Exception as e:
        st.error(f"Sistem başlatılırken hata oluştu: {str(e)}")
        logger.error(f"System initialization error: {str(e)}")

if __name__ == "__main__":
    main()
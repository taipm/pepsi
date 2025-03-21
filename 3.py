# Import thêm các thư viện để nâng cao chất lượng
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import TokenTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import json
import time
from datetime import datetime
import hashlib
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import threading

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EducationalRAG")

class EducationalRAG:
    def __init__(self, 
                 pdf_dir="books/", 
                 embedding_model="dangvantuan/vietnamese-embedding",
                 llm_model="mrjacktung/phogpt-4b-chat-gguf",
                 vector_db_path="vector_store",
                 metadata_path="metadata.json",
                 cache_dir="document_cache",
                 vector_db_type="faiss",  # hoặc "chroma"
                 chunk_size=1000,
                 chunk_overlap=200,
                 k_retrieval=5,
                 temperature=0):
        
        self.pdf_dir = pdf_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_db_path = vector_db_path
        self.metadata_path = metadata_path
        self.cache_dir = cache_dir
        self.vector_db_type = vector_db_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.temperature = temperature
        
        # Đảm bảo các thư mục tồn tại
        for dir_path in [pdf_dir, cache_dir, vector_db_path]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Khởi tạo metadata nếu chưa tồn tại
        if not os.path.exists(metadata_path):
            self.metadata = {
                "files": {}, 
                "last_update": None,
                "config": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": embedding_model,
                    "vector_db_type": vector_db_type
                }
            }
            self._save_metadata()
        else:
            self._load_metadata()
            
            # Kiểm tra cấu hình
            if self._is_config_changed():
                logger.warning("Cấu hình đã thay đổi. Có thể cần tạo lại vector store.")
        
        # Khởi tạo các thành phần
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Hỗ trợ nhiều embedding models khác nhau
        self.embeddings = self._init_embeddings()
        
        # Khởi tạo LLM
        self.llm = ChatOllama(model=llm_model, temperature=temperature)
        
        # Tải hoặc tạo mới vector store
        self.vector_store = None
        self._init_vector_store()
        
        # Lock để đồng bộ hóa cập nhật
        self.update_lock = threading.Lock()
    
    def _init_embeddings(self):
        """Khởi tạo embedding model"""
        try:
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo embedding model: {e}")
            logger.info("Đang thử sử dụng embedding model mặc định...")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def _is_config_changed(self):
        """Kiểm tra xem cấu hình có thay đổi so với lần chạy trước không"""
        current_config = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "vector_db_type": self.vector_db_type
        }
        
        saved_config = self.metadata.get("config", {})
        
        # Cập nhật config nếu chưa có
        if not saved_config:
            self.metadata["config"] = current_config
            self._save_metadata()
            return False
        
        return any(current_config[key] != saved_config.get(key) for key in current_config)
        
    def _load_metadata(self):
        """Tải metadata từ file"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            # Thêm config nếu chưa có
            if "config" not in self.metadata:
                self.metadata["config"] = {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": self.embedding_model,
                    "vector_db_type": self.vector_db_type
                }
                self._save_metadata()
        except Exception as e:
            logger.error(f"Lỗi khi tải metadata: {e}")
            self.metadata = {"files": {}, "last_update": None, "config": {}}
            
    def _save_metadata(self):
        """Lưu metadata vào file"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu metadata: {e}")
            
    def _calculate_file_hash(self, file_path):
        """Tính toán hash của file để kiểm tra thay đổi"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)  # Đọc theo từng khối để tiết kiệm bộ nhớ
                while buf:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Lỗi khi tính hash cho file {file_path}: {e}")
            return None
    
    def _init_vector_store(self):
        """Khởi tạo vector store, tải từ đĩa nếu đã tồn tại"""
        vector_store_exists = False
        
        if self.vector_db_type == "faiss":
            vector_store_exists = os.path.exists(f"{self.vector_db_path}/index.faiss")
            
            if vector_store_exists:
                try:
                    logger.info("Đang tải FAISS vector store từ đĩa...")
                    self.vector_store = FAISS.load_local(
                        folder_path=self.vector_db_path, 
                        embeddings=self.embeddings
                    )
                    logger.info("Đã tải FAISS vector store thành công!")
                except Exception as e:
                    logger.error(f"Lỗi khi tải FAISS vector store: {e}")
                    vector_store_exists = False
        
        elif self.vector_db_type == "chroma":
            vector_store_exists = os.path.exists(f"{self.vector_db_path}/chroma.sqlite3")
            
            if vector_store_exists:
                try:
                    logger.info("Đang tải Chroma vector store từ đĩa...")
                    self.vector_store = Chroma(
                        persist_directory=self.vector_db_path,
                        embedding_function=self.embeddings
                    )
                    logger.info("Đã tải Chroma vector store thành công!")
                except Exception as e:
                    logger.error(f"Lỗi khi tải Chroma vector store: {e}")
                    vector_store_exists = False
        
        if not vector_store_exists:
            logger.info("Chưa có vector store, sẽ tạo mới khi thêm tài liệu...")
            self.vector_store = None
    
    def _get_document_cache_path(self, file_path):
        """Tạo đường dẫn file cache cho tài liệu đã xử lý"""
        file_name = os.path.basename(file_path)
        return os.path.join(self.cache_dir, f"{file_name}.pickle")
    
    def update_knowledge_base(self, force_reload=False):
        """Cập nhật cơ sở kiến thức bằng cách quét thư mục PDF và cập nhật nếu cần"""
        with self.update_lock:  # Sử dụng lock để đảm bảo chỉ một quy trình cập nhật chạy cùng lúc
            logger.info(f"Đang quét thư mục {self.pdf_dir} để tìm tài liệu mới hoặc đã thay đổi...")
            
            has_updates = False
            all_documents = []
            
            try:
                pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            except Exception as e:
                logger.error(f"Lỗi khi quét thư mục {self.pdf_dir}: {e}")
                return False
            
            # Kiểm tra các file đã không còn tồn tại
            removed_files = [f for f in self.metadata["files"] if f not in pdf_files]
            for removed in removed_files:
                logger.info(f"File {removed} đã bị xóa, loại bỏ khỏi metadata")
                del self.metadata["files"][removed]
                has_updates = True
            
            # Xử lý từng file PDF
            for pdf in pdf_files:
                try:
                    file_path = os.path.join(self.pdf_dir, pdf)
                    file_hash = self._calculate_file_hash(file_path)
                    
                    if file_hash is None:
                        logger.warning(f"Bỏ qua file {pdf} do không thể tính hash")
                        continue
                        
                    file_modified = os.path.getmtime(file_path)
                    
                    cache_path = self._get_document_cache_path(file_path)
                    
                    # Kiểm tra nếu file đã thay đổi hoặc chưa được xử lý
                    if (pdf not in self.metadata["files"] or 
                        self.metadata["files"][pdf]["hash"] != file_hash or
                        force_reload):
                        
                        logger.info(f"Đang xử lý tài liệu mới/đã thay đổi: {pdf}")
                        
                        # Tải và xử lý tài liệu
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        
                        # Thêm metadata chi tiết hơn
                        for doc in docs:
                            doc.metadata["source"] = pdf
                            doc.metadata["source_path"] = file_path
                            doc.metadata["page"] = doc.metadata.get("page", "")
                            doc.metadata["extracted_at"] = datetime.now().isoformat()
                            doc.metadata["file_hash"] = file_hash
                            
                            # Làm sạch nội dung
                            doc.page_content = self._clean_text(doc.page_content)
                        
                        # Lưu vào cache để tái sử dụng
                        with open(cache_path, 'wb') as f:
                            pickle.dump(docs, f)
                        
                        # Cập nhật metadata
                        self.metadata["files"][pdf] = {
                            "hash": file_hash,
                            "last_modified": file_modified,
                            "last_processed": time.time(),
                            "path": file_path,
                            "total_pages": len(docs)
                        }
                        
                        all_documents.extend(docs)
                        has_updates = True
                        logger.info(f"Đã xử lý {len(docs)} trang từ {pdf}")
                    else:
                        logger.info(f"Tài liệu {pdf} không thay đổi, đang tải từ cache...")
                        if os.path.exists(cache_path):
                            with open(cache_path, 'rb') as f:
                                cached_docs = pickle.load(f)
                                all_documents.extend(cached_docs)
                        else:
                            logger.warning(f"Cache cho {pdf} không tồn tại. Đang xử lý lại...")
                            loader = PyPDFLoader(file_path)
                            docs = loader.load()
                            
                            # Thêm metadata chi tiết hơn
                            for doc in docs:
                                doc.metadata["source"] = pdf
                                doc.metadata["source_path"] = file_path
                                doc.metadata["page"] = doc.metadata.get("page", "")
                                doc.metadata["extracted_at"] = datetime.now().isoformat()
                                
                                # Làm sạch nội dung
                                doc.page_content = self._clean_text(doc.page_content)
                            
                            # Lưu vào cache
                            with open(cache_path, 'wb') as cf:
                                pickle.dump(docs, cf)
                                
                            all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý file {pdf}: {e}")
            
            # Nếu có cập nhật hoặc chưa có vector store, tạo mới vector store
            if has_updates or self.vector_store is None:
                logger.info("Đang cập nhật vector store...")
                
                # Chia nhỏ văn bản
                logger.info("Đang chia nhỏ văn bản...")
                split_documents = self.text_splitter.split_documents(all_documents)
                logger.info(f"Đã chia thành {len(split_documents)} đoạn văn bản.")
                
                # Loại bỏ đoạn văn bản trùng lặp
                split_documents = self._remove_duplicate_chunks(split_documents)
                logger.info(f"Còn lại {len(split_documents)} đoạn sau khi loại bỏ trùng lặp.")
                
                # Tạo hoặc cập nhật vector store
                if self.vector_store is None:
                    logger.info("Tạo mới vector store...")
                    
                    if self.vector_db_type == "faiss":
                        self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
                        self.vector_store.save_local(self.vector_db_path)
                    elif self.vector_db_type == "chroma":
                        self.vector_store = Chroma.from_documents(
                            documents=split_documents,
                            embedding=self.embeddings,
                            persist_directory=self.vector_db_path
                        )
                        self.vector_store.persist()
                else:
                    logger.info("Thêm tài liệu mới vào vector store...")
                    self.vector_store.add_documents(split_documents)
                    
                    # Lưu vector store
                    if self.vector_db_type == "faiss":
                        self.vector_store.save_local(self.vector_db_path)
                    elif self.vector_db_type == "chroma":
                        self.vector_store.persist()
                
                logger.info("Đã lưu vector store.")
                
                # Cập nhật metadata
                self.metadata["last_update"] = time.time()
                self._save_metadata()
                logger.info("Đã cập nhật metadata.")
            else:
                logger.info("Không có tài liệu mới hoặc thay đổi.")
            
            return has_updates
    
    def _clean_text(self, text):
        """Làm sạch văn bản từ PDF"""
        if not text:
            return ""
            
        # Loại bỏ các ký tự không phù hợp, khoảng trắng thừa
        text = ' '.join(text.split())
        
        # Xử lý các ký tự đặc biệt
        text = text.replace('\x0c', ' ').replace('\t', ' ')
        
        return text
    
    def _remove_duplicate_chunks(self, documents, similarity_threshold=0.85):
        """Loại bỏ các đoạn văn bản quá giống nhau dựa trên embedding similarity"""
        try:
            # Nếu ít hơn 5 văn bản, giữ nguyên
            if len(documents) < 5:
                return documents
                
            # Sử dụng EmbeddingsRedundantFilter
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=self.embeddings,
                similarity_threshold=similarity_threshold
            )
            
            filtered_docs = redundant_filter.transform_documents(documents)
            return filtered_docs
        except Exception as e:
            logger.error(f"Lỗi khi loại bỏ đoạn trùng lặp: {e}")
            # Trả về danh sách gốc nếu có lỗi
            return documents
    
    def setup_qa_chain(self):
        """Thiết lập chain RAG với tham chiếu đến nguồn"""
        # Đảm bảo vector store đã được khởi tạo
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy thêm tài liệu trước.")
        
        # Thiết lập retriever cơ bản
        base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k_retrieval},
            search_type="similarity"
        )
        
        # Thiết lập retriever với nén ngữ cảnh để cải thiện kết quả
        try:
            # Sử dụng bộ lọc embedding để loại bỏ kết quả không liên quan
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.7
            )
            
            # Thiết lập pipeline nén tài liệu
            pipeline = DocumentCompressorPipeline(
                transformers=[embeddings_filter]
            )
            
            # Tạo retriever với nén
            retriever = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=base_retriever
            )
            
            logger.info("Đã thiết lập retriever với nén ngữ cảnh")
        except Exception as e:
            logger.warning(f"Không thể thiết lập retriever với nén: {e}")
            logger.info("Sử dụng retriever cơ bản")
            retriever = base_retriever
        
        # Custom prompt với yêu cầu trích dẫn nguồn
        prompt_template = """
        Bạn là trợ lý học tập thông minh với khả năng trả lời các câu hỏi dựa trên tài liệu.
        
        ### Câu hỏi:
        {question}
        
        ### Thông tin từ tài liệu:
        {context}
        
        ### Hướng dẫn:
        - Trả lời dựa trên thông tin từ tài liệu được cung cấp.
        - Nếu không tìm thấy thông tin, hãy thừa nhận điều đó thay vì đưa ra thông tin không chính xác.
        - Phân tích thông tin từ nhiều nguồn tài liệu nếu có.
        - Kết thúc câu trả lời bằng cách liệt kê các nguồn tài liệu tham khảo với format:
          [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
        - Đảm bảo câu trả lời súc tích, rõ ràng và dễ hiểu.
        - Nếu câu hỏi yêu cầu so sánh hoặc phân tích, hãy tổ chức câu trả lời một cách logic.
        
        ### Câu trả lời:
        """
        
        # Tạo prompt
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Chuẩn bị chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def answer_question(self, question):
        """Trả lời câu hỏi sử dụng RAG chain"""
        # Đảm bảo có vector store
        if self.vector_store is None:
            logger.info("Chưa có tài liệu nào. Đang thử cập nhật cơ sở kiến thức...")
            self.update_knowledge_base()
            
            # Kiểm tra lại sau khi cập nhật
            if self.vector_store is None:
                return "Không tìm thấy tài liệu nào để trả lời. Vui lòng thêm tài liệu trước."
        
        # Tạo chain
        try:
            qa_chain = self.setup_qa_chain()
            
            # Đo thời gian truy vấn
            start_time = time.time()
            
            # Thực hiện truy vấn
            answer = qa_chain.invoke(question)
            
            query_time = time.time() - start_time
            logger.info(f"Thời gian truy vấn: {query_time:.2f} giây")
            
            return answer
        except Exception as e:
            logger.error(f"Lỗi khi trả lời câu hỏi: {e}")
            return f"Đã xảy ra lỗi khi trả lời câu hỏi. Vui lòng thử lại. Chi tiết: {str(e)}"
    
    def get_similar_questions(self, question, n=3):
        """Gợi ý các câu hỏi tương tự dựa trên câu hỏi hiện tại"""
        if not hasattr(self, 'question_bank') or not self.question_bank:
            # Tạo ngân hàng câu hỏi mẫu dựa trên nội dung tài liệu
            self.question_bank = [
                "Phân biệt giữa thời tiết và khí hậu.",
                "Đặc điểm của các đới khí hậu chính trên Trái Đất là gì?",
                "Nguyên nhân gây ra hiện tượng El Nino là gì?",
                "Sự khác biệt giữa bão và áp thấp nhiệt đới là gì?",
                "Các yếu tố ảnh hưởng đến khí hậu Việt Nam là gì?",
                "Tại sao miền Bắc Việt Nam có 4 mùa rõ rệt?",
                "Hiện tượng biến đổi khí hậu ảnh hưởng như thế nào đến Việt Nam?",
                "Đặc điểm của gió mùa Đông Bắc ở Việt Nam?",
                "Tại sao miền Nam Việt Nam chỉ có hai mùa mưa và khô?",
                "Đặc điểm khí hậu vùng núi cao ở Việt Nam có gì đặc biệt?"
            ]
        
        # Lấy embedding của câu hỏi hiện tại
        try:
            question_embedding = self.embeddings.embed_query(question)
            
            # Tính similarity với các câu hỏi trong ngân hàng
            similarities = []
            for q in self.question_bank:
                q_embedding = self.embeddings.embed_query(q)
                similarity = self._cosine_similarity(question_embedding, q_embedding)
                similarities.append((q, similarity))
            
            # Sắp xếp theo độ tương đồng giảm dần
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Trả về n câu hỏi tương tự nhất (loại bỏ câu giống hệt)
            similar_questions = [q for q, sim in similarities if q.lower() != question.lower()][:n]
            return similar_questions
        except Exception as e:
            logger.error(f"Lỗi khi tìm câu hỏi tương tự: {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Tính độ tương đồng cosine giữa hai vector"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
    
    def get_statistics(self):
        """Trả về thống kê về dữ liệu đã xử lý"""
        stats = {
            "total_documents": len(self.metadata["files"]),
            "total_pages": sum(file_info.get("total_pages", 0) for file_info in self.metadata["files"].values()),
            "last_update": datetime.fromtimestamp(self.metadata["last_update"]).strftime('%Y-%m-%d %H:%M:%S') if self.metadata["last_update"] else None,
            "documents": [{"filename": f, "pages": info.get("total_pages", 0)} for f, info in self.metadata["files"].items()],
            "vector_db_type": self.vector_db_type,
            "embedding_model": self.embedding_model
        }
        return stats
    
    def evaluate_retrieval(self, test_questions, ground_truth_answers, k=5):
        """Đánh giá chất lượng của hệ thống truy xuất"""
        if self.vector_store is None:
            logger.error("Vector store chưa được khởi tạo")
            return {"error": "Vector store chưa được khởi tạo"}
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        results = []
        for q, a in zip(test_questions, ground_truth_answers):
            try:
                # Lấy các đoạn văn bản liên quan
                docs = retriever.get_relevant_documents(q)
                
                # Kiểm tra xem văn bản có chứa câu trả lời không
                contains_answer = any(a.lower() in doc.page_content.lower() for doc in docs)
                
                # Tính điểm trung bình của các đoạn văn bản
                score = sum(1 for doc in docs if a.lower() in doc.page_content.lower()) / len(docs) if docs else 0
                
                results.append({
                    "question": q,
                    "contains_answer": contains_answer,
                    "score": score,
                    "num_relevant_docs": len(docs)
                })
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá câu hỏi '{q}': {e}")
                results.append({
                    "question": q,
                    "error": str(e)
                })
        
        # Tính toán tổng hợp
        success_rate = sum(1 for r in results if r.get("contains_answer", False)) / len(results) if results else 0
        avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
        
        return {
            "success_rate": success_rate,
            "average_score": avg_score,
            "details": results
        }

# Hàm main có thêm nhiều tùy chọn để chạy chương trình
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hệ thống RAG cho tài liệu giáo dục")
    parser.add_argument("--pdf_dir", default="books/", help="Thư mục chứa tài liệu PDF")
    parser.add_argument("--vector_db", default="faiss", choices=["faiss", "chroma"], help="Loại vector database")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Kích thước chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Độ chồng lấp của các chunk")
    parser.add_argument("--force_update", action="store_true", help="Buộc cập nhật lại tất cả tài liệu")
    parser.add_argument("--evaluate", action="store_true", help="Chạy đánh giá hệ thống")
    
    args = parser.parse_args()
    
    # Khởi tạo EducationalRAG với các tham số từ dòng lệnh
    rag = EducationalRAG(
        pdf_dir=args.pdf_dir,
        vector_db_type=args.vector_db,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Cập nhật cơ sở kiến thức
    rag.update_knowledge_base(force_reload=args.force_update)
    
    # In thống kê
    stats = rag.get_statistics()
    print("\nThống kê:")
    print(f"Tổng số tài liệu: {stats['total_documents']}")
    print(f"Tổng số trang: {stats['total_pages']}")
    print(f"Cập nhật lần cuối: {stats['last_update']}")
    print(f"Vector DB: {stats['vector_db_type']}")
    print(f"Embedding model: {stats['embedding_model']}\n")
    
    # Chạy đánh giá nếu được yêu cầu
    if args.evaluate:
        print("Đang đánh giá hệ thống...")
        test_questions = [
            "Phân biệt giữa thời tiết và khí hậu.",
            "Đặc điểm của các đới khí hậu chính trên Trái Đất là gì?"
        ]
        ground_truth = [
            "thời tiết", 
            "nhiệt đới"
        ]
        eval_results = rag.evaluate_retrieval(test_questions, ground_truth)
        print(f"Tỷ lệ thành công: {eval_results['success_rate']:.2f}")
        print(f"Điểm trung bình: {eval_results['average_score']:.2f}")
    
    # Chạy interactive mode để người dùng có thể hỏi câu hỏi
    print("=== Trợ lý học tập ===")
    print("Nhập 'exit' để thoát, 'update' để cập nhật cơ sở kiến thức.\n")
    
    while True:
        question = input("\nNhập câu hỏi của bạn: ")
        
        if question.lower() == 'exit':
            break
        elif question.lower() == 'update':
            print("\nĐang cập nhật cơ sở kiến thức...")
            rag.update_knowledge_base(force_reload=True)
            print("Đã cập nhật xong!")
            continue
        elif question.lower() == 'stats':
            stats = rag.get_statistics()
            print("\nThống kê:")
            for key, value in stats.items():
                if key != "documents":
                    print(f"{key}: {value}")
            continue
        
        print("\nĐang tìm câu trả lời...")
        answer = rag.answer_question(question)
        print(f"\nCâu trả lời: {answer}")
        
        # Gợi ý câu hỏi liên quan
        similar_questions = rag.get_similar_questions(question)
        if similar_questions:
            print("\nCâu hỏi liên quan bạn có thể quan tâm:")
            for i, q in enumerate(similar_questions, 1):
                print(f"{i}. {q}")

if __name__ == "__main__":
    main()
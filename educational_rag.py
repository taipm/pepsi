import os
import json
import time
import hashlib
import base64
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough

from utils import ImageExtractor, QualityEvaluator
from memory_manager import ConversationMemoryManager

class EducationalRAG:
    def __init__(self, pdf_dir="books/", vector_db_path="vector_store"):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.metadata_path = "metadata.json"
        self.conversations_path = "conversations"
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs("document_cache", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        os.makedirs(self.conversations_path, exist_ok=True)
        
        # Khởi tạo các thành phần
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
        self.llm = ChatOllama(model="mrjacktung/phogpt-4b-chat-gguf", temperature=0)
        
        # Tải hoặc khởi tạo metadata
        self.metadata = self._load_or_create_metadata()
        
        # Tải vector store
        self.vector_store = self._load_vector_store()
        
        # Khởi tạo các thành phần mới
        self.image_extractor = ImageExtractor()
        self.quality_evaluator = QualityEvaluator(self.llm)
        self.memory_manager = ConversationMemoryManager(self.conversations_path)
        
        # Biến tạm lưu session hiện tại
        self.current_session_id = None
        self.current_memory = None
    
    def _load_or_create_metadata(self):
        """Tải metadata hoặc tạo mới nếu chưa tồn tại"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"files": {}, "last_update": None, "images": {}}
    
    def _save_metadata(self):
        """Lưu metadata vào file"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _load_vector_store(self):
        """Tải vector store nếu đã tồn tại"""
        if os.path.exists(f"{self.vector_db_path}.faiss") and os.path.exists(f"{self.vector_db_path}.pkl"):
            try:
                print(f"Đang tải vector store từ {self.vector_db_path}...")
                start_time = time.time()
                vector_store = FAISS.load_local(self.vector_db_path, self.embeddings)
                load_time = time.time() - start_time
                print(f"Đã tải vector store trong {load_time:.2f} giây")
                return vector_store
            except Exception as e:
                print(f"Lỗi khi tải vector store: {e}")
                print("Sẽ tạo vector store mới khi cập nhật tài liệu.")
                return None
        return None
    
    def _calculate_file_hash(self, file_path):
        """Tính toán hash của file để kiểm tra thay đổi"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def update_knowledge_base(self, force_reload=False):
        """Cập nhật cơ sở kiến thức khi có tài liệu mới"""
        print(f"Đang quét thư mục {self.pdf_dir} để tìm tài liệu mới hoặc đã thay đổi...")
        
        has_updates = False
        all_documents = []
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        # Kiểm tra nếu không có file PDF nào
        if not pdf_files:
            print(f"Không tìm thấy file PDF nào trong thư mục {self.pdf_dir}")
            return False
        
        for pdf in pdf_files:
            file_path = os.path.join(self.pdf_dir, pdf)
            file_hash = self._calculate_file_hash(file_path)
            
            # Kiểm tra nếu file đã thay đổi hoặc chưa được xử lý
            if (pdf not in self.metadata["files"] or 
                self.metadata["files"][pdf]["hash"] != file_hash or
                force_reload):
                
                print(f"Đang xử lý tài liệu mới/đã thay đổi: {pdf}")
                
                # Tải và xử lý tài liệu
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Thêm metadata chi tiết
                for doc in docs:
                    doc.metadata["source"] = pdf
                    doc.metadata["page"] = doc.metadata.get("page", "")
                    doc.metadata["timestamp"] = datetime.now().isoformat()
                
                # Trích xuất hình ảnh từ PDF
                print(f"Đang trích xuất hình ảnh từ {pdf}...")
                try:
                    extracted_images = self.image_extractor.extract_images_from_pdf(file_path)
                    # Lưu thông tin về hình ảnh vào metadata
                    if pdf not in self.metadata["images"]:
                        self.metadata["images"][pdf] = {}
                    
                    for page_num, images in extracted_images.items():
                        self.metadata["images"][pdf][str(page_num)] = []
                        for i, img_data in enumerate(images):
                            img_filename = f"{pdf.replace('.pdf', '')}_page{page_num}_img{i}.png"
                            img_path = os.path.join("images", img_filename)
                            img_data.save(img_path)
                            
                            self.metadata["images"][pdf][str(page_num)].append({
                                "filename": img_filename,
                                "path": img_path,
                                "width": img_data.width,
                                "height": img_data.height
                            })
                    
                    print(f"Đã trích xuất {sum(len(imgs) for imgs in extracted_images.values())} hình ảnh từ {pdf}")
                except Exception as e:
                    print(f"Lỗi khi trích xuất hình ảnh từ {pdf}: {e}")
                
                # Cache tài liệu đã xử lý
                cache_path = os.path.join("document_cache", f"{pdf}.json")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump([{"content": doc.page_content, "metadata": doc.metadata} for doc in docs], f, ensure_ascii=False)
                
                # Cập nhật metadata
                self.metadata["files"][pdf] = {
                    "hash": file_hash,
                    "last_modified": os.path.getmtime(file_path),
                    "last_processed": time.time(),
                    "path": file_path,
                    "total_pages": len(docs)
                }
                
                all_documents.extend(docs)
                has_updates = True
                print(f"Đã xử lý {len(docs)} trang từ {pdf}")
            else:
                # Nếu file không thay đổi, nạp từ cache để có đủ tài liệu cho vector store
                print(f"Tài liệu {pdf} không thay đổi, đang tải từ cache...")
                cache_path = os.path.join("document_cache", f"{pdf}.json")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached_data = json.load(f)
                            cached_docs = []
                            for item in cached_data:
                                doc = Document(page_content=item["content"], metadata=item["metadata"])
                                cached_docs.append(doc)
                            all_documents.extend(cached_docs)
                    except Exception as e:
                        print(f"Lỗi khi tải cache cho {pdf}: {e}")
        
        # Xử lý khi có tài liệu
        if all_documents:
            # Chia nhỏ văn bản
            print("Đang chia nhỏ văn bản...")
            split_documents = self.text_splitter.split_documents(all_documents)
            print(f"Đã chia thành {len(split_documents)} đoạn văn bản.")
            
            # Tạo hoặc cập nhật vector store
            if has_updates or self.vector_store is None:
                print("Đang cập nhật vector store...")
                
                if not split_documents:
                    print("Không có đoạn văn bản nào sau khi chia. Bỏ qua cập nhật vector store.")
                    return False
                
                start_time = time.time()
                if self.vector_store is None:
                    print("Tạo mới vector store...")
                    self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
                else:
                    print("Thêm tài liệu mới vào vector store...")
                    self.vector_store.add_documents(split_documents)
                
                # Lưu vector store
                self.vector_store.save_local(self.vector_db_path)
                indexing_time = time.time() - start_time
                print(f"Đã cập nhật vector store trong {indexing_time:.2f} giây")
                
                # Cập nhật metadata
                self.metadata["last_update"] = time.time()
                self._save_metadata()
                
                return True
            
            print("Vector store đã được tải và không cần cập nhật.")
            return True
        
        print("Không có tài liệu nào được xử lý.")
        return False

    def start_new_session(self, session_name=None):
        """Bắt đầu phiên trò chuyện mới và trả về ID phiên"""
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = self.memory_manager.create_new_session(session_name)
        self.current_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print(f"Đã tạo phiên mới với ID: {self.current_session_id}")
        return self.current_session_id
    
    def load_session(self, session_id):
        """Tải phiên trò chuyện đã lưu"""
        if self.memory_manager.session_exists(session_id):
            self.current_session_id = session_id
            conversation_data = self.memory_manager.load_session(session_id)
            
            # Khởi tạo memory mới
            self.current_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Thêm các tin nhắn từ phiên đã lưu vào memory
            for exchange in conversation_data["exchanges"]:
                self.current_memory.chat_memory.add_user_message(exchange["question"])
                self.current_memory.chat_memory.add_ai_message(exchange["answer"])
            
            print(f"Đã tải phiên {session_id} với {len(conversation_data['exchanges'])} lượt trao đổi")
            return True
        else:
            print(f"Không tìm thấy phiên {session_id}")
            return False
    
    def list_sessions(self):
        """Liệt kê các phiên trò chuyện đã lưu"""
        return self.memory_manager.list_sessions()
    
    def create_qa_chain(self):
        """Tạo chain QA với tham chiếu nguồn và lịch sử hội thoại"""
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy cập nhật cơ sở kiến thức trước.")
        
        # Thiết lập retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Tùy chỉnh prompt với yêu cầu trích dẫn nguồn và hỗ trợ lịch sử trò chuyện
        template = """
        Bạn là trợ lý học tập thông minh dựa trên tài liệu sách giáo khoa.
        
        ### Lịch sử trò chuyện:
        {chat_history}
        
        ### Câu hỏi hiện tại:
        {question}
        
        ### Thông tin từ tài liệu:
        {context}
        
        ### Hướng dẫn:
        1. Trả lời dựa trên thông tin từ tài liệu được cung cấp.
        2. Xem xét lịch sử trò chuyện để hiểu ngữ cảnh và đảm bảo tính nhất quán.
        3. Nếu không tìm thấy thông tin, hãy thừa nhận điều đó thay vì đưa ra thông tin không chính xác.
        4. Nếu câu hỏi liên quan đến hình ảnh được đề cập trong tài liệu, hãy đề xuất hiển thị hình ảnh đó.
        5. Phân tích các dữ liệu từ nhiều nguồn khác nhau để đưa ra câu trả lời toàn diện.
        6. Sắp xếp câu trả lời theo cấu trúc rõ ràng và dễ hiểu.
        7. Kết thúc câu trả lời bằng cách liệt kê các nguồn tài liệu tham khảo với format:
           [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
        
        ### Trả lời:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question", "context"]
        )
        
        # Tạo chain QA với memory
        if self.current_memory is None:
            # Khởi tạo memory mới nếu chưa có
            self.start_new_session()
        
        # Kết hợp retrieval QA với memory
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough(), "chat_history": lambda x: self.current_memory.load_memory_variables({})["chat_history"]}
            | prompt
            | self.llm
        )
        
        return qa_chain
    
    def answer_question(self, question):
        """Trả lời câu hỏi với tham chiếu nguồn và hỗ trợ lịch sử trò chuyện"""
        if self.vector_store is None:
            print("Chưa có tài liệu nào. Đang cập nhật cơ sở kiến thức...")
            success = self.update_knowledge_base(force_reload=True)
            
            if not success or self.vector_store is None:
                print("Không thể tạo vector store. Đang thử tạo lại từ đầu...")
                # Xóa metadata để buộc xử lý lại toàn bộ tài liệu
                self.metadata = {"files": {}, "last_update": None, "images": {}}
                self._save_metadata()
                success = self.update_knowledge_base(force_reload=True)
                
                if not success or self.vector_store is None:
                    return {"answer": "Không thể tạo vector store. Vui lòng kiểm tra lại tài liệu và thư mục.", "images": [], "quality_score": 0}
        
        # Đảm bảo phiên hiện tại đã được tạo
        if self.current_session_id is None:
            self.start_new_session()
        
        # Tạo chain QA
        try:
            qa_chain = self.create_qa_chain()
            
            # Đo thời gian truy vấn
            start_time = time.time()
            
            # Thực hiện truy vấn 
            result = qa_chain.invoke(question)
            answer = result.content
            
            # Lấy nguồn tài liệu tham khảo từ câu trả lời
            source_docs = self._extract_source_references(answer)
            
            # Tìm kiếm hình ảnh liên quan trong tài liệu
            images = self._find_relevant_images(source_docs)
            
            # Đánh giá chất lượng câu trả lời
            quality_score = self.quality_evaluator.evaluate_answer(question, answer, source_docs)
            
            # Lưu câu hỏi và câu trả lời vào bộ nhớ
            self.current_memory.save_context(
                {"input": question},
                {"output": answer}
            )
            
            # Lưu phiên trò chuyện hiện tại
            if self.current_session_id:
                self.memory_manager.save_exchange(
                    self.current_session_id,
                    question,
                    answer,
                    source_docs,
                    images,
                    quality_score
                )
            
            query_time = time.time() - start_time
            print(f"Thời gian truy vấn: {query_time:.2f} giây")
            
            # Trả về câu trả lời, hình ảnh liên quan và điểm đánh giá
            return {
                "answer": answer,
                "images": images,
                "quality_score": quality_score,
                "source_docs": source_docs
            }
            
        except Exception as e:
            print(f"Lỗi khi trả lời câu hỏi: {str(e)}")
            error_message = f"Đã xảy ra lỗi khi tìm câu trả lời: {str(e)}. Vui lòng thử lại hoặc cập nhật cơ sở kiến thức."
            return {"answer": error_message, "images": [], "quality_score": 0, "source_docs": []}
    
    def process_feedback(self, question, answer, feedback, feedback_text=None):
        """Xử lý phản hồi từ người dùng để học chủ động"""
        if self.current_session_id is None:
            print("Không có phiên nào đang hoạt động.")
            return False
        
        print(f"Nhận phản hồi: {feedback} - {feedback_text}")
        
        # Lưu phản hồi vào phiên hiện tại
        self.memory_manager.add_feedback(
            self.current_session_id,
            question,
            answer,
            feedback,
            feedback_text
        )
        
        # Phân tích phản hồi để học hỏi
        if feedback == "negative" and feedback_text:
            # Tạo prompt để phân tích lỗi và cải thiện
            improvement_analysis = self._analyze_improvement(question, answer, feedback_text)
            
            # Lưu phân tích để sử dụng trong tương lai
            self.memory_manager.save_learning(
                self.current_session_id,
                question,
                answer,
                feedback_text,
                improvement_analysis
            )
            
            print("Đã phân tích và lưu bài học từ phản hồi")
            return True
        
        return True
    
    def _analyze_improvement(self, question, answer, feedback):
        """Phân tích phản hồi và đề xuất cách cải thiện"""
        prompt = f"""
        Hãy phân tích câu hỏi, câu trả lời và phản hồi của người dùng để tìm ra cách cải thiện:
        
        Câu hỏi: {question}
        
        Câu trả lời: {answer}
        
        Phản hồi người dùng: {feedback}
        
        Yêu cầu:
        1. Chỉ ra vấn đề chính trong câu trả lời
        2. Giải thích cách trả lời tốt hơn
        3. Đề xuất cách cải thiện quy trình tìm kiếm hoặc tổng hợp thông tin
        
        Kết quả phân tích:
        """
        
        try:
            analysis_result = self.llm.invoke(prompt).content
            return analysis_result
        except Exception as e:
            print(f"Lỗi khi phân tích cải thiện: {e}")
            return "Không thể phân tích phản hồi do lỗi."
    
    def _extract_source_references(self, answer):
        """Trích xuất thông tin nguồn tham khảo từ câu trả lời"""
        source_docs = []
        lines = answer.split('\n')
        
        for line in lines:
            if "[Nguồn tham khảo:" in line:
                # Tách thông tin nguồn
                try:
                    source_info = line.strip()[line.find("[Nguồn tham khảo:") + len("[Nguồn tham khảo:"):].strip()
                    if source_info.endswith("]"):
                        source_info = source_info[:-1]
                    
                    parts = source_info.split(", trang")
                    if len(parts) == 2:
                        source = parts[0].strip()
                        page = parts[1].strip()
                        source_docs.append({"source": source, "page": page})
                except:
                    continue
        
        return source_docs
    
    def _find_relevant_images(self, source_docs):
        """Tìm hình ảnh liên quan dựa trên các nguồn tài liệu được tham chiếu"""
        relevant_images = []
        
        for doc in source_docs:
            source = doc.get("source")
            page = doc.get("page")
            
            if source and page and source in self.metadata["images"]:
                # Kiểm tra xem có hình ảnh cho trang này không
                if page in self.metadata["images"][source]:
                    for img_info in self.metadata["images"][source][page]:
                        # Đọc hình ảnh và mã hóa base64
                        try:
                            with open(img_info["path"], "rb") as img_file:
                                img_data = img_file.read()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                
                                relevant_images.append({
                                    "filename": img_info["filename"],
                                    "source": source,
                                    "page": page,
                                    "data": img_base64
                                })
                        except Exception as e:
                            print(f"Lỗi khi đọc hình ảnh {img_info['path']}: {e}")
        
        return relevant_images
    
    def get_statistics(self):
        """Trả về thống kê về dữ liệu đã xử lý"""
        stats = {
            "total_documents": len(self.metadata["files"]),
            "total_pages": sum(file_info.get("total_pages", 0) for file_info in self.metadata["files"].values()),
            "last_update": datetime.fromtimestamp(self.metadata["last_update"]).strftime('%Y-%m-%d %H:%M:%S') if self.metadata["last_update"] else None,
            "documents": [{"filename": f, "pages": info.get("total_pages", 0)} for f, info in self.metadata["files"].items()],
            "total_images": sum(sum(len(imgs) for imgs in doc.values()) for doc in self.metadata["images"].values()) if self.metadata.get("images") else 0,
            "total_sessions": len(self.memory_manager.list_sessions()),
        }
        
        if self.current_session_id:
            current_session = self.memory_manager.load_session(self.current_session_id)
            stats["current_session"] = {
                "id": self.current_session_id,
                "name": current_session.get("name", ""),
                "exchanges": len(current_session.get("exchanges", [])),
                "created_at": current_session.get("created_at", ""),
            }
        
        return stats
    
    def get_learning_insights(self):
        """Lấy thông tin insights từ quá trình học hỏi"""
        return self.memory_manager.get_learning_insights()
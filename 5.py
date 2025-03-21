import os
import json
import time
import hashlib
import base64
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("educational_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EducationalRAG")

class ConversationMemory:
    """Lưu trữ và quản lý lịch sử hội thoại"""
    
    def __init__(self, memory_file="conversation_history.json", max_history=50):
        self.memory_file = memory_file
        self.max_history = max_history
        self.conversations = self._load_or_create_memory()
        self.current_session_id = None
        
    def _load_or_create_memory(self) -> Dict:
        """Tải lịch sử hội thoại hoặc tạo mới nếu chưa tồn tại"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Lỗi khi tải file lịch sử: {e}")
                return {"sessions": {}}
        return {"sessions": {}}
    
    def _save_memory(self):
        """Lưu lịch sử hội thoại vào file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu lịch sử: {e}")
    
    def create_new_session(self) -> str:
        """Tạo phiên hội thoại mới"""
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.conversations["sessions"][session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "messages": []
        }
        self.current_session_id = session_id
        self._save_memory()
        return session_id
    
    def add_message(self, question: str, answer: str, 
                   sources: List[Dict] = None, 
                   feedback: Dict = None,
                   images: List[Dict] = None) -> None:
        """Thêm tin nhắn vào phiên hội thoại hiện tại"""
        if not self.current_session_id:
            self.create_new_session()
            
        message = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "feedback": feedback or {},
            "images": images or []
        }
        
        self.conversations["sessions"][self.current_session_id]["messages"].append(message)
        self.conversations["sessions"][self.current_session_id]["last_updated"] = datetime.now().isoformat()
        
        # Giới hạn số lượng tin nhắn lưu trữ
        if len(self.conversations["sessions"][self.current_session_id]["messages"]) > self.max_history:
            self.conversations["sessions"][self.current_session_id]["messages"] = \
                self.conversations["sessions"][self.current_session_id]["messages"][-self.max_history:]
                
        self._save_memory()
    
    def add_feedback(self, message_index: int, feedback_data: Dict) -> None:
        """Thêm đánh giá cho một câu trả lời cụ thể"""
        if not self.current_session_id:
            logger.warning("Không có phiên hội thoại nào đang hoạt động")
            return
            
        try:
            messages = self.conversations["sessions"][self.current_session_id]["messages"]
            if 0 <= message_index < len(messages):
                messages[message_index]["feedback"] = feedback_data
                messages[message_index]["feedback"]["feedback_time"] = datetime.now().isoformat()
                self._save_memory()
                return True
            else:
                logger.warning(f"Không tìm thấy tin nhắn với chỉ số {message_index}")
                return False
        except Exception as e:
            logger.error(f"Lỗi khi thêm đánh giá: {e}")
            return False
    
    def get_history(self, session_id: str = None, limit: int = None) -> List[Dict]:
        """Lấy lịch sử hội thoại của một phiên cụ thể"""
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.conversations["sessions"]:
            return []
            
        messages = self.conversations["sessions"][session_id]["messages"]
        if limit:
            return messages[-limit:]
        return messages
    
    def get_session_list(self) -> List[Dict]:
        """Lấy danh sách các phiên hội thoại"""
        return [
            {
                "session_id": session_id,
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated"),
                "message_count": len(session_data.get("messages", []))
            }
            for session_id, session_data in self.conversations["sessions"].items()
        ]
    
    def load_session(self, session_id: str) -> bool:
        """Tải một phiên hội thoại cụ thể"""
        if session_id in self.conversations["sessions"]:
            self.current_session_id = session_id
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Xóa một phiên hội thoại cụ thể"""
        if session_id in self.conversations["sessions"]:
            del self.conversations["sessions"][session_id]
            self._save_memory()
            
            # Reset current session nếu phiên hiện tại bị xóa
            if self.current_session_id == session_id:
                self.current_session_id = None
                
            return True
        return False
    
    def get_messages_as_langchain_format(self, limit: int = 5) -> List:
        """Chuyển đổi lịch sử tin nhắn sang định dạng cho LangChain"""
        if not self.current_session_id:
            return []
            
        messages = self.get_history(limit=limit)
        langchain_messages = []
        
        for msg in messages:
            langchain_messages.append(HumanMessage(content=msg["question"]))
            langchain_messages.append(AIMessage(content=msg["answer"]))
            
        return langchain_messages
    
    def get_recent_questions(self, limit: int = 5) -> List[str]:
        """Lấy danh sách các câu hỏi gần đây"""
        if not self.current_session_id:
            return []
            
        messages = self.get_history(limit=limit)
        return [msg["question"] for msg in messages]
    
    def get_chat_context(self, limit: int = 5) -> str:
        """Tạo ngữ cảnh hội thoại để làm đầu vào cho LLM"""
        if not self.current_session_id:
            return ""
            
        messages = self.get_history(limit=limit)
        context = ""
        
        for i, msg in enumerate(messages):
            context += f"Người dùng: {msg['question']}\n"
            context += f"Trợ lý: {msg['answer']}\n\n"
            
        return context.strip()


class ImageExtractor:
    """Trích xuất và quản lý hình ảnh từ tài liệu PDF"""
    
    def __init__(self, image_dir="extracted_images"):
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        
    # def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
    #     """Trích xuất hình ảnh từ file PDF và trả về thông tin"""
    #     from PyPDF2 import PdfReader
    #     from PIL import Image
    #     import io
        
    #     logger.info(f"Đang trích xuất hình ảnh từ {pdf_path}")
        
    #     # Tạo thư mục dành riêng cho file này
    #     pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    #     pdf_image_dir = os.path.join(self.image_dir, pdf_name)
    #     os.makedirs(pdf_image_dir, exist_ok=True)
        
    #     # Đọc PDF
    #     pdf = PdfReader(pdf_path)
    #     images_info = []
        
    #     for i, page in enumerate(pdf.pages):
    #         page_num = i + 1
            
    #         # Lấy resources của trang
    #         if '/Resources' in page:
    #             resources = page['/Resources']
    #             if '/XObject' in resources:
    #                 xobject = resources['/XObject']
                    
    #                 # Lấy các objects (có thể là hình ảnh)
    #                 for obj_name in xobject:
    #                     obj = xobject[obj_name]
                        
    #                     # Nếu là hình ảnh
    #                     if obj['/Subtype'] == '/Image':
    #                         try:
    #                             # Trích xuất dữ liệu hình ảnh
    #                             data = obj.get_data()
                                
    #                             # Xác định định dạng hình ảnh
    #                             if '/Filter' in obj:
    #                                 filters = obj['/Filter']
                                    
    #                                 # Xử lý JPG
    #                                 if filters == '/DCTDecode' or (isinstance(filters, list) and '/DCTDecode' in filters):
    #                                     img_format = 'jpg'
    #                                 # Xử lý PNG
    #                                 elif filters == '/FlateDecode' or (isinstance(filters, list) and '/FlateDecode' in filters):
    #                                     img_format = 'png'
    #                                 else:
    #                                     img_format = 'jpg'  # Default format
    #                             else:
    #                                 img_format = 'jpg'
                                
    #                             # Tạo tên file
    #                             image_filename = f"{pdf_name}_page{page_num}_{len(images_info)}.{img_format}"
    #                             image_path = os.path.join(pdf_image_dir, image_filename)
                                
    #                             # Lưu hình ảnh
    #                             with open(image_path, 'wb') as img_file:
    #                                 img_file.write(data)
                                
    #                             # Lấy kích thước hình ảnh
    #                             with Image.open(image_path) as img:
    #                                 width, height = img.size
                                
    #                             # Lưu thông tin hình ảnh
    #                             image_info = {
    #                                 "source": pdf_path,
    #                                 "page": page_num,
    #                                 "path": image_path,
    #                                 "format": img_format,
    #                                 "width": width,
    #                                 "height": height,
    #                                 "filename": image_filename
    #                             }
                                
    #                             images_info.append(image_info)
    #                             logger.info(f"Đã trích xuất hình ảnh: {image_filename}")
                                
    #                         except Exception as e:
    #                             logger.error(f"Lỗi khi trích xuất hình ảnh: {e}")
        
    #     logger.info(f"Đã trích xuất tổng cộng {len(images_info)} hình ảnh từ {pdf_path}")
    #     return images_info
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Trích xuất hình ảnh từ file PDF và trả về thông tin"""
        try:
            from PyPDF2 import PdfReader
            import fitz  # PyMuPDF - cần cài đặt: pip install PyMuPDF
            import io
            from PIL import Image
            
            logger.info(f"Đang trích xuất hình ảnh từ {pdf_path}")
            
            # Tạo thư mục dành riêng cho file này
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            pdf_image_dir = os.path.join(self.image_dir, pdf_name)
            os.makedirs(pdf_image_dir, exist_ok=True)
            
            images_info = []
            
            # Phương pháp 1: Sử dụng PyMuPDF (hiệu quả hơn)
            try:
                doc = fitz.open(pdf_path)
                image_count = 0
                
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    page_num = page_index + 1
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Tạo tên file và lưu hình ảnh
                        image_filename = f"{pdf_name}_page{page_num}_{image_count}.{image_ext}"
                        image_path = os.path.join(pdf_image_dir, image_filename)
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Lấy kích thước hình ảnh
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            width, height = img.size
                        
                        # Lưu thông tin hình ảnh
                        image_info = {
                            "source": pdf_path,
                            "page": page_num,
                            "path": image_path,
                            "format": image_ext,
                            "width": width,
                            "height": height,
                            "filename": image_filename
                        }
                        
                        images_info.append(image_info)
                        image_count += 1
                
                if images_info:
                    logger.info(f"Đã trích xuất {len(images_info)} hình ảnh từ {pdf_path} (PyMuPDF)")
                    return images_info
            except Exception as e:
                logger.error(f"Lỗi khi trích xuất hình ảnh với PyMuPDF: {e}")
            
            # Phương pháp 2: Sử dụng PyPDF2 (nếu phương pháp 1 thất bại)
            pdf = PdfReader(pdf_path)
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Lấy resources của trang
                if '/Resources' in page:
                    resources = page['/Resources']
                    if '/XObject' in resources:
                        xobject = resources['/XObject']
                        
                        # Lấy các objects (có thể là hình ảnh)
                        for obj_name in xobject:
                            obj = xobject[obj_name]
                            
                            # Nếu là hình ảnh
                            if obj['/Subtype'] == '/Image':
                                try:
                                    # Trích xuất dữ liệu hình ảnh
                                    data = obj.get_data()
                                    
                                    # Xác định định dạng hình ảnh
                                    if '/Filter' in obj:
                                        filters = obj['/Filter']
                                        
                                        # Xử lý JPG
                                        if filters == '/DCTDecode' or (isinstance(filters, list) and '/DCTDecode' in filters):
                                            img_format = 'jpg'
                                        # Xử lý PNG
                                        elif filters == '/FlateDecode' or (isinstance(filters, list) and '/FlateDecode' in filters):
                                            img_format = 'png'
                                        else:
                                            img_format = 'jpg'  # Default format
                                    else:
                                        img_format = 'jpg'
                                    
                                    # Tạo tên file
                                    image_filename = f"{pdf_name}_page{page_num}_{len(images_info)}.{img_format}"
                                    image_path = os.path.join(pdf_image_dir, image_filename)
                                    
                                    # Lưu hình ảnh
                                    with open(image_path, 'wb') as img_file:
                                        img_file.write(data)
                                    
                                    # Lấy kích thước hình ảnh
                                    with Image.open(image_path) as img:
                                        width, height = img.size
                                    
                                    # Lưu thông tin hình ảnh
                                    image_info = {
                                        "source": pdf_path,
                                        "page": page_num,
                                        "path": image_path,
                                        "format": img_format,
                                        "width": width,
                                        "height": height,
                                        "filename": image_filename
                                    }
                                    
                                    images_info.append(image_info)
                                    logger.info(f"Đã trích xuất hình ảnh: {image_filename}")
                                    
                                except Exception as e:
                                    logger.error(f"Lỗi khi trích xuất hình ảnh: {e}")
            
            logger.info(f"Đã trích xuất tổng cộng {len(images_info)} hình ảnh từ {pdf_path}")
            return images_info
            
        except Exception as e:
            logger.error(f"Lỗi không xác định khi trích xuất hình ảnh: {e}")
            return []
    
    def get_image_base64(self, image_path: str) -> str:
        """Chuyển đổi hình ảnh sang base64 để hiển thị"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi hình ảnh sang base64: {e}")
            return ""
    
    def find_relevant_images(self, query: str, page_numbers: List[int], pdf_name: str = None) -> List[Dict]:
        """Tìm hình ảnh liên quan đến câu hỏi dựa trên số trang"""
        relevant_images = []
        
        pdf_name = pdf_name or ""
        
        # Tìm trong tất cả thư mục hình ảnh
        if not pdf_name:
            for dir_name in os.listdir(self.image_dir):
                dir_path = os.path.join(self.image_dir, dir_name)
                if os.path.isdir(dir_path):
                    for img_name in os.listdir(dir_path):
                        # Kiểm tra xem hình ảnh có thuộc các trang liên quan không
                        for page in page_numbers:
                            if f"page{page}_" in img_name:
                                img_path = os.path.join(dir_path, img_name)
                                try:
                                    with Image.open(img_path) as img:
                                        width, height = img.size
                                    
                                    relevant_images.append({
                                        "path": img_path,
                                        "filename": img_name,
                                        "page": page,
                                        "source": dir_name,
                                        "width": width,
                                        "height": height,
                                        "base64": self.get_image_base64(img_path)
                                    })
                                except Exception as e:
                                    logger.error(f"Lỗi khi xử lý hình ảnh {img_path}: {e}")
        # Tìm trong thư mục hình ảnh của một PDF cụ thể
        else:
            dir_path = os.path.join(self.image_dir, pdf_name)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for img_name in os.listdir(dir_path):
                    for page in page_numbers:
                        if f"page{page}_" in img_name:
                            img_path = os.path.join(dir_path, img_name)
                            try:
                                with Image.open(img_path) as img:
                                    width, height = img.size
                                
                                relevant_images.append({
                                    "path": img_path,
                                    "filename": img_name,
                                    "page": page,
                                    "source": pdf_name,
                                    "width": width,
                                    "height": height,
                                    "base64": self.get_image_base64(img_path)
                                })
                            except Exception as e:
                                logger.error(f"Lỗi khi xử lý hình ảnh {img_path}: {e}")
        
        return relevant_images


class FeedbackAnalyzer:
    """Phân tích và học từ phản hồi người dùng"""
    
    def __init__(self, feedback_file="feedback_data.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_or_create_feedback()
        
    def _load_or_create_feedback(self) -> Dict:
        """Tải dữ liệu đánh giá hoặc tạo mới nếu chưa có"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Lỗi khi tải dữ liệu đánh giá: {e}")
                return self._create_new_feedback_data()
        return self._create_new_feedback_data()
    
    def _create_new_feedback_data(self) -> Dict:
        """Tạo cấu trúc dữ liệu đánh giá mới"""
        return {
            "feedbacks": [],
            "metrics": {
                "total_feedbacks": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "avg_rating": 0,
                "topic_ratings": {}
            },
            "learning_patterns": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_feedback(self):
        """Lưu dữ liệu đánh giá vào file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu đánh giá: {e}")
    
    def add_feedback(self, question: str, answer: str, feedback: Dict) -> None:
        """Thêm đánh giá mới và cập nhật chỉ số"""
        feedback_entry = {
            "id": len(self.feedback_data["feedbacks"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "rating": feedback.get("rating", 3),  # Thang điểm 1-5
            "correct": feedback.get("correct", True),
            "helpful": feedback.get("helpful", True),
            "missing_info": feedback.get("missing_info", False),
            "comment": feedback.get("comment", ""),
            "suggested_answer": feedback.get("suggested_answer", ""),
            "categories": feedback.get("categories", []),
            "sources": feedback.get("sources", [])
        }
        
        # Thêm vào danh sách
        self.feedback_data["feedbacks"].append(feedback_entry)
        
        # Cập nhật chỉ số
        self._update_metrics(feedback_entry)
        
        # Cập nhật mẫu học
        self._update_learning_patterns(feedback_entry)
        
        # Lưu lại
        self._save_feedback()
    
    def _update_metrics(self, feedback_entry: Dict) -> None:
        """Cập nhật các chỉ số tổng hợp"""
        metrics = self.feedback_data["metrics"]
        metrics["total_feedbacks"] += 1
        
        # Phân loại đánh giá
        rating = feedback_entry["rating"]
        if rating >= 4:
            metrics["positive_count"] += 1
        elif rating <= 2:
            metrics["negative_count"] += 1
        else:
            metrics["neutral_count"] += 1
        
        # Cập nhật điểm trung bình
        total_ratings = metrics["positive_count"] + metrics["neutral_count"] + metrics["negative_count"]
        metrics["avg_rating"] = (metrics["positive_count"] * 5 + metrics["neutral_count"] * 3 + metrics["negative_count"]) / total_ratings
        
        # Cập nhật điểm theo chủ đề
        for category in feedback_entry["categories"]:
            if category not in metrics["topic_ratings"]:
                metrics["topic_ratings"][category] = {"count": 0, "avg_rating": 0}
            
            topic_data = metrics["topic_ratings"][category]
            topic_data["count"] += 1
            topic_data["avg_rating"] = ((topic_data["count"] - 1) * topic_data["avg_rating"] + rating) / topic_data["count"]
    
    def _update_learning_patterns(self, feedback_entry: Dict) -> None:
        """Cập nhật mẫu học từ phản hồi"""
        patterns = self.feedback_data["learning_patterns"]
        question = feedback_entry["question"]
        
        # Phân tích từ khóa từ câu hỏi
        keywords = self._extract_keywords(question)
        
        # Cập nhật mẫu học cho từng từ khóa
        for keyword in keywords:
            if keyword not in patterns:
                patterns[keyword] = {
                    "count": 0,
                    "avg_rating": 0,
                    "correct_count": 0,
                    "helpful_count": 0,
                    "missing_info_count": 0,
                    "examples": []
                }
                
            pattern = patterns[keyword]
            pattern["count"] += 1
            
            # Cập nhật thống kê
            old_avg = pattern["avg_rating"]
            pattern["avg_rating"] = ((pattern["count"] - 1) * old_avg + feedback_entry["rating"]) / pattern["count"]
            
            if feedback_entry["correct"]:
                pattern["correct_count"] += 1
                
            if feedback_entry["helpful"]:
                pattern["helpful_count"] += 1
                
            if feedback_entry["missing_info"]:
                pattern["missing_info_count"] += 1
            
            # Lưu ví dụ (giới hạn số lượng)
            if len(pattern["examples"]) < 5:
                pattern["examples"].append({
                    "question": question,
                    "rating": feedback_entry["rating"],
                    "id": feedback_entry["id"]
                })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa từ văn bản"""
        # Phiên bản đơn giản - tách từ và lọc những từ > 3 ký tự
        words = text.lower().split()
        # Lọc stopwords (có thể thêm danh sách stopwords tiếng Việt)
        stopwords = ["của", "là", "và", "có", "trong", "cho", "các", "với", "được", "những"]
        return [w for w in words if len(w) > 3 and w not in stopwords]
    
    def get_quality_assessment(self, question: str, answer: str) -> Dict:
        """Đánh giá chất lượng câu trả lời dựa trên dữ liệu đã học"""
        # Trích xuất từ khóa
        keywords = self._extract_keywords(question)
        
        # Tính toán điểm chất lượng dự đoán
        if not keywords or not self.feedback_data["learning_patterns"]:
            return {
                "predicted_rating": 3.0,
                "confidence": 0.0,
                "missing_info_probability": 0.0,
                "similar_questions": []
            }
        
        # Tính toán chỉ số từ các từ khóa
        total_weight = 0
        weighted_rating = 0
        missing_info_probability = 0
        similar_questions = []
        
        for keyword in keywords:
            if keyword in self.feedback_data["learning_patterns"]:
                pattern = self.feedback_data["learning_patterns"][keyword]
                weight = pattern["count"] / 10  # Chuẩn hóa trọng số
                if weight > 1:
                    weight = 1
                
                weighted_rating += pattern["avg_rating"] * weight
                missing_info_probability += (pattern["missing_info_count"] / pattern["count"]) * weight
                total_weight += weight
                
                # Thêm các câu hỏi tương tự
                for example in pattern["examples"]:
                    similar_questions.append(example)
        
        # Tính toán kết quả cuối cùng
        if total_weight > 0:
            predicted_rating = weighted_rating / total_weight
            missing_info_probability = missing_info_probability / total_weight
            confidence = min(1.0, total_weight / 3)  # 3 từ khóa với trọng số 1.0 sẽ có độ tin cậy 100%
        else:
            predicted_rating = 3.0
            missing_info_probability = 0.0
            confidence = -0.0
            
        # Sắp xếp và lọc câu hỏi tương tự
        similar_questions = sorted(similar_questions, key=lambda x: x["rating"], reverse=True)
        similar_questions = similar_questions[:3]  # Giới hạn 3 câu hỏi
            
        return {
            "predicted_rating": round(predicted_rating, 1),
            "confidence": round(confidence, 2),
            "missing_info_probability": round(missing_info_probability, 2),
            "similar_questions": similar_questions
        }
    
    def get_feedback_statistics(self) -> Dict:
        """Lấy thống kê về dữ liệu đánh giá"""
        return self.feedback_data["metrics"]
    
    def get_insights(self) -> Dict:
        """Trích xuất insights từ dữ liệu đánh giá"""
        metrics = self.feedback_data["metrics"]
        patterns = self.feedback_data["learning_patterns"]
        
        # Xác định từ khóa có điểm thấp nhất
        problematic_keywords = []
        for keyword, data in patterns.items():
            if data["count"] >= 3 and data["avg_rating"] < 3:
                problematic_keywords.append({
                    "keyword": keyword,
                    "avg_rating": data["avg_rating"],
                    "count": data["count"],
                    "missing_info_count": data["missing_info_count"]
                })
        
        # Sắp xếp theo điểm thấp nhất
        problematic_keywords = sorted(problematic_keywords, key=lambda x: x["avg_rating"])[:5]
        
        # Xác định chủ đề có điểm thấp nhất
        problematic_topics = []
        for topic, data in metrics["topic_ratings"].items():
            if data["count"] >= 3 and data["avg_rating"] < 3:
                problematic_topics.append({
                    "topic": topic,
                    "avg_rating": data["avg_rating"],
                    "count": data["count"]
                })
        
        # Sắp xếp theo điểm thấp nhất
        problematic_topics = sorted(problematic_topics, key=lambda x: x["avg_rating"])[:3]
        
        return {
            "total_feedbacks": metrics["total_feedbacks"],
            "avg_rating": metrics["avg_rating"],
            "problematic_keywords": problematic_keywords,
            "problematic_topics": problematic_topics,
            "positive_rate": metrics["positive_count"] / metrics["total_feedbacks"] if metrics["total_feedbacks"] > 0 else 0
        }


class EducationalRAG:
    def __init__(self, pdf_dir="books/", vector_db_path="vector_store"):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.metadata_path = "metadata.json"
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs("document_cache", exist_ok=True)
        
        # Khởi tạo các thành phần
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
        self.llm = ChatOllama(model="mrjacktung/phogpt-4b-chat-gguf", temperature=0)
        
        # Tải hoặc khởi tạo metadata
        self.metadata = self._load_or_create_metadata()
        
        # Tải vector store
        self.vector_store = self._load_vector_store()
        
        # Khởi tạo các thành phần nâng cao
        self.conversation_memory = ConversationMemory()
        self.image_extractor = ImageExtractor()
        self.feedback_analyzer = FeedbackAnalyzer()
        
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
                logger.info(f"Đang tải vector store từ {self.vector_db_path}...")
                start_time = time.time()
                vector_store = FAISS.load_local(self.vector_db_path, self.embeddings)
                load_time = time.time() - start_time
                logger.info(f"Đã tải vector store trong {load_time:.2f} giây")
                return vector_store
            except Exception as e:
                logger.error(f"Lỗi khi tải vector store: {e}")
                logger.info("Sẽ tạo vector store mới khi cập nhật tài liệu.")
                return None
        return None
    
    def validate_answer(self, answer: str) -> str:
        """Kiểm tra và sửa lỗi trong câu trả lời"""
        import re

        # Kiểm tra xem câu trả lời có phải là danh sách câu hỏi không
        question_patterns = [
            r"^\d+\.\s+[A-Z][^.?!]*\?",  # Dạng "1. Câu hỏi?"
            r"^[A-Z][^.?!]*\?",          # Dạng "Câu hỏi?"
            r"^-\s+[A-Z][^.?!]*\?"       # Dạng "- Câu hỏi?"
        ]

        is_question_list = False
        for pattern in question_patterns:
            if re.search(pattern, answer, re.MULTILINE):
                is_question_list = True
                break

        if is_question_list:
            message = """
            Tôi không tìm thấy đủ thông tin để trả lời câu hỏi này một cách đầy đủ từ các tài liệu hiện có. 
            
            Để có thể cung cấp câu trả lời chính xác về sự phân biệt giữa thời tiết và khí hậu cùng với ví dụ minh họa, tôi cần thêm tài liệu hoặc thông tin chi tiết hơn.
            
            Vui lòng thử cập nhật cơ sở kiến thức hoặc sử dụng câu hỏi cụ thể hơn.
            """
            return message.strip()

        return answer
    
    def _calculate_file_hash(self, file_path):
        """Tính toán hash của file để kiểm tra thay đổi"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def update_knowledge_base(self, force_reload=False):
        """Cập nhật cơ sở kiến thức khi có tài liệu mới"""
        logger.info(f"Đang quét thư mục {self.pdf_dir} để tìm tài liệu mới hoặc đã thay đổi...")
        
        has_updates = False
        all_documents = []
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        # Kiểm tra nếu không có file PDF nào
        if not pdf_files:
            logger.warning(f"Không tìm thấy file PDF nào trong thư mục {self.pdf_dir}")
            return False
        
        for pdf in pdf_files:
            file_path = os.path.join(self.pdf_dir, pdf)
            file_hash = self._calculate_file_hash(file_path)
            
            # Kiểm tra nếu file đã thay đổi hoặc chưa được xử lý
            if (pdf not in self.metadata["files"] or 
                self.metadata["files"][pdf]["hash"] != file_hash or
                force_reload):
                
                logger.info(f"Đang xử lý tài liệu mới/đã thay đổi: {pdf}")
                
                # Tải và xử lý tài liệu
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Thêm metadata chi tiết
                for doc in docs:
                    doc.metadata["source"] = pdf
                    doc.metadata["page"] = doc.metadata.get("page", "")
                    doc.metadata["timestamp"] = datetime.now().isoformat()
                
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
                
                # Trích xuất hình ảnh nếu chưa có
                if pdf not in self.metadata.get("images", {}):
                    logger.info(f"Đang trích xuất hình ảnh từ {pdf}...")
                    extracted_images = self.image_extractor.extract_images_from_pdf(file_path)
                    self.metadata.setdefault("images", {})[pdf] = {
                        "count": len(extracted_images),
                        "last_extracted": time.time(),
                        "images": [img["filename"] for img in extracted_images]
                    }
                
                all_documents.extend(docs)
                has_updates = True
                logger.info(f"Đã xử lý {len(docs)} trang từ {pdf}")
            else:
                # Nếu file không thay đổi, nạp từ cache để có đủ tài liệu cho vector store
                logger.info(f"Tài liệu {pdf} không thay đổi, đang tải từ cache...")
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
                        logger.error(f"Lỗi khi tải cache cho {pdf}: {e}")
        
        # Xử lý khi có tài liệu
        if all_documents:
            # Chia nhỏ văn bản
            logger.info("Đang chia nhỏ văn bản...")
            split_documents = self.text_splitter.split_documents(all_documents)
            logger.info(f"Đã chia thành {len(split_documents)} đoạn văn bản.")
            
            # Tạo hoặc cập nhật vector store
            if has_updates or self.vector_store is None:
                logger.info("Đang cập nhật vector store...")
                
                if not split_documents:
                    logger.warning("Không có đoạn văn bản nào sau khi chia. Bỏ qua cập nhật vector store.")
                    return False
                
                start_time = time.time()
                if self.vector_store is None:
                    logger.info("Tạo mới vector store...")
                    self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
                else:
                    logger.info("Thêm tài liệu mới vào vector store...")
                    self.vector_store.add_documents(split_documents)
                
                # Lưu vector store
                self.vector_store.save_local(self.vector_db_path)
                indexing_time = time.time() - start_time
                logger.info(f"Đã cập nhật vector store trong {indexing_time:.2f} giây")
                
                # Cập nhật metadata
                self.metadata["last_update"] = time.time()
                self._save_metadata()
                
                return True
            
            logger.info("Vector store đã được tải và không cần cập nhật.")
            return True
        
        logger.info("Không có tài liệu nào được xử lý.")
        return False
    
    # def create_qa_chain(self, use_conversation_history=True):
    #     """Tạo chain QA với tham chiếu nguồn và lịch sử hội thoại"""
    #     if self.vector_store is None:
    #         raise ValueError("Vector store chưa được khởi tạo. Hãy cập nhật cơ sở kiến thức trước.")
        
    #     # Thiết lập retriever
    #     retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
    #     # Tùy chỉnh prompt với yêu cầu trích dẫn nguồn
    #     if use_conversation_history:
    #         # Prompt với lịch sử hội thoại
    #         context = self.conversation_memory.get_chat_context(limit=3)
    #         template = f"""
    #         Bạn là trợ lý học tập thông minh dựa trên tài liệu sách giáo khoa.
            
    #         ### Lịch sử hội thoại:
    #         {context}
            
    #         ### Câu hỏi hiện tại:
    #         {{question}}
            
    #         ### Thông tin từ tài liệu:
    #         {{context}}
            
    #         ### Hướng dẫn:
    #         1. Trả lời dựa trên thông tin từ tài liệu được cung cấp.
    #         2. Nếu không tìm thấy thông tin, hãy thừa nhận điều đó thay vì đưa ra thông tin không chính xác.
    #         3. Phân tích các dữ liệu từ nhiều nguồn khác nhau để đưa ra câu trả lời toàn diện.
    #         4. Sắp xếp câu trả lời theo cấu trúc rõ ràng và dễ hiểu.
    #         5. Tham khảo lịch sử hội thoại để đảm bảo nhất quán nhưng ưu tiên thông tin từ tài liệu.
    #         6. Kết thúc câu trả lời bằng cách liệt kê các nguồn tài liệu tham khảo với format:
    #            [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
            
    #         ### Trả lời:
    #         """
    #     else:
    #         # Prompt không có lịch sử hội thoại
    #         template = """
    #         Bạn là trợ lý học tập thông minh dựa trên tài liệu sách giáo khoa.
            
    #         ### Câu hỏi:
    #         {question}
            
    #         ### Thông tin từ tài liệu:
    #         {context}
            
    #         ### Hướng dẫn:
    #         1. Trả lời dựa trên thông tin từ tài liệu được cung cấp.
    #         2. Nếu không tìm thấy thông tin, hãy thừa nhận điều đó thay vì đưa ra thông tin không chính xác.
    #         3. Phân tích các dữ liệu từ nhiều nguồn khác nhau để đưa ra câu trả lời toàn diện.
    #         4. Sắp xếp câu trả lời theo cấu trúc rõ ràng và dễ hiểu.
    #         5. Kết thúc câu trả lời bằng cách liệt kê các nguồn tài liệu tham khảo với format:
    #            [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
            
    #         ### Trả lời:
    #         """
        
    #     prompt = PromptTemplate(
    #         template=template,
    #         input_variables=["question", "context"]
    #     )
        
    #     # Tạo chain QA
    #     qa_chain = RetrievalQA.from_chain_type(
    #         llm=self.llm,
    #         chain_type="stuff",
    #         retriever=retriever,
    #         chain_type_kwargs={"prompt": prompt},
    #         return_source_documents=True
    #     )
        
    #     return qa_chain
    def create_qa_chain(self, use_conversation_history=True):
        """Tạo chain QA với tham chiếu nguồn và lịch sử hội thoại"""
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy cập nhật cơ sở kiến thức trước.")
        
        # Thiết lập retriever - tăng số lượng đoạn văn để có thông tin tốt hơn
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Tùy chỉnh prompt với hướng dẫn rõ ràng hơn
        template = """
        Bạn là trợ lý học tập thông minh dựa trên tài liệu sách giáo khoa.
        
        NHIỆM VỤ: Cung cấp câu trả lời CHÍNH XÁC và ĐẦY ĐỦ cho câu hỏi dưới đây, dựa trên thông tin từ tài liệu.
        
        ### Câu hỏi:
        {question}
        
        ### Thông tin từ tài liệu:
        {context}
        
        ### Hướng dẫn chi tiết:
        1. Trả lời trực tiếp câu hỏi, KHÔNG liệt kê các câu hỏi khác hoặc tạo danh sách câu hỏi.
        2. Dựa hoàn toàn vào thông tin từ tài liệu được cung cấp.
        3. Nếu không tìm thấy thông tin, hãy thừa nhận: "Tôi không tìm thấy thông tin đầy đủ về điều này trong tài liệu."
        4. Câu trả lời phải CÓ CẤU TRÚC RÕ RÀNG, ĐẦY ĐỦ và TOÀN DIỆN.
        5. Kết thúc bằng cách liệt kê nguồn tham khảo với format: [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
        
        ### Trả lời:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )
        
        # Tạo chain QA với temperature thấp hơn để đảm bảo sự nhất quán
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOllama(model="mrjacktung/phogpt-4b-chat-gguf", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def answer_question(self, question, use_conversation_history=True, with_feedback=True):
        """Trả lời câu hỏi với tham chiếu nguồn, lịch sử hội thoại và phản hồi chất lượng"""
        if self.vector_store is None:
            logger.warning("Chưa có tài liệu nào. Đang cập nhật cơ sở kiến thức...")
            success = self.update_knowledge_base(force_reload=True)
            
            if not success or self.vector_store is None:
                logger.warning("Không thể tạo vector store. Đang thử tạo lại từ đầu...")
                # Xóa metadata để buộc xử lý lại toàn bộ tài liệu
                self.metadata = {"files": {}, "last_update": None, "images": {}}
                self._save_metadata()
                success = self.update_knowledge_base(force_reload=True)
                
                if not success or self.vector_store is None:
                    return {"answer": "Không thể tạo vector store. Vui lòng kiểm tra lại tài liệu và thư mục.",
                            "images": [], "sources": [], "feedback": None}
        
        # Tạo chain QA
        try:
            qa_chain = self.create_qa_chain(use_conversation_history=use_conversation_history)
            
            # Đo thời gian truy vấn
            start_time = time.time()
            
            # Thực hiện truy vấn
            result = qa_chain({"query": question})
            
            # Xử lý kết quả để đảm bảo có tham chiếu
            # answer = result["result"]
             # Sau khi lấy kết quả từ LLM, thêm phần xác thực
            answer = self.validate_answer(result["result"])
            source_docs = result["source_documents"]
            
            # Kiểm tra nếu câu trả lời đã có tham chiếu
            if "[Nguồn tham khảo:" not in answer:
                # Nếu chưa có, thêm tham chiếu
                references = []
                for doc in source_docs:
                    source = doc.metadata.get("source", "Không rõ nguồn")
                    page = doc.metadata.get("page", "")
                    ref = f"[Nguồn tham khảo: {source}, trang {page}]"
                    if ref not in references:
                        references.append(ref)
                
                # Thêm tham chiếu vào câu trả lời
                if references:
                    answer += "\n\n" + "\n".join(references)
            
            # Lấy danh sách các trang được tham chiếu để tìm hình ảnh
            page_numbers = []
            sources = []
            for doc in source_docs:
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page", "")
                
                try:
                    if page:
                        page_num = int(page)
                        if page_num not in page_numbers:
                            page_numbers.append(page_num)
                            
                    if source and {"source": source, "page": page} not in sources:
                        sources.append({"source": source, "page": page})
                except ValueError:
                    pass
            
            # Tìm hình ảnh liên quan
            relevant_images = []
            for source in sources:
                pdf_name = source.get("source", "").replace(".pdf", "")
                page = source.get("page", "")
                
                if pdf_name and page:
                    try:
                        page_num = int(page)
                        images = self.image_extractor.find_relevant_images([page_num], pdf_name)
                        relevant_images.extend(images)
                    except ValueError:
                        pass
            
            query_time = time.time() - start_time
            logger.info(f"Thời gian truy vấn: {query_time:.2f} giây")
            
            # Đánh giá chất lượng câu trả lời
            quality_assessment = None
            if with_feedback:
                quality_assessment = self.feedback_analyzer.get_quality_assessment(question, answer)
            
            # Thêm vào lịch sử hội thoại
            self.conversation_memory.add_message(
                question=question,
                answer=answer,
                sources=sources,
                images=[img["path"] for img in relevant_images]
            )
            
            return {
                "answer": answer,
                "images": relevant_images,
                "sources": sources,
                "feedback": quality_assessment
            }
        except Exception as e:
            logger.error(f"Lỗi khi trả lời câu hỏi: {str(e)}")
            return {
                "answer": f"Đã xảy ra lỗi khi tìm câu trả lời: {str(e)}. Vui lòng thử lại hoặc cập nhật cơ sở kiến thức.",
                "images": [],
                "sources": [],
                "feedback": None
            }
    
    def add_feedback(self, question, answer, feedback_data):
        """Thêm đánh giá của người dùng"""
        # Thêm vào bộ phân tích đánh giá
        self.feedback_analyzer.add_feedback(question, answer, feedback_data)
        
        # Cập nhật đánh giá trong lịch sử hội thoại
        if self.conversation_memory.current_session_id:
            history = self.conversation_memory.get_history()
            if history:
                for i, message in enumerate(reversed(history)):
                    if message["question"] == question and message["answer"] == answer:
                        self.conversation_memory.add_feedback(len(history) - i - 1, feedback_data)
                        break
        
        return True
    
    def get_statistics(self):
        """Trả về thống kê về dữ liệu đã xử lý"""
        stats = {
            "total_documents": len(self.metadata["files"]),
            "total_pages": sum(file_info.get("total_pages", 0) for file_info in self.metadata["files"].values()),
            "last_update": datetime.fromtimestamp(self.metadata["last_update"]).strftime('%Y-%m-%d %H:%M:%S') if self.metadata["last_update"] else None,
            "documents": [{"filename": f, "pages": info.get("total_pages", 0)} for f, info in self.metadata["files"].items()],
            "total_images": sum(img_info.get("count", 0) for img_info in self.metadata.get("images", {}).values()),
            "conversation_sessions": len(self.conversation_memory.conversations["sessions"]),
            "feedback_count": self.feedback_analyzer.get_feedback_statistics()["total_feedbacks"]
        }
        return stats
    
    def get_conversation_sessions(self):
        """Lấy danh sách các phiên hội thoại"""
        return self.conversation_memory.get_session_list()
    
    def load_conversation_session(self, session_id):
        """Tải một phiên hội thoại cụ thể"""
        return self.conversation_memory.load_session(session_id)
    
    def create_new_conversation(self):
        """Tạo phiên hội thoại mới"""
        return self.conversation_memory.create_new_session()
    
    def get_feedback_insights(self):
        """Lấy insights từ dữ liệu đánh giá"""
        return self.feedback_analyzer.get_insights()


# Hàm hiển thị console cải tiến
def display_rich_answer(answer_data):
    """Hiển thị câu trả lời với định dạng phong phú"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import box
    from rich.columns import Columns
    
    console = Console()
    
    # Hiển thị câu trả lời chính
    answer = answer_data["answer"]
    console.print(Panel(Markdown(answer), title="Câu trả lời", border_style="green"))
    
    # Hiển thị đánh giá chất lượng nếu có
    feedback = answer_data.get("feedback")
    if feedback:
        feedback_table = Table(show_header=False, box=box.ROUNDED)
        feedback_table.add_column("Tiêu chí", style="cyan")
        feedback_table.add_column("Giá trị", style="yellow")
        
        feedback_table.add_row("Điểm dự đoán", f"{feedback['predicted_rating']}/5.0")
        feedback_table.add_row("Độ tin cậy", f"{feedback['confidence']*100:.0f}%")
        feedback_table.add_row("Khả năng thiếu thông tin", f"{feedback['missing_info_probability']*100:.0f}%")
        
        console.print(Panel(feedback_table, title="Đánh giá chất lượng", border_style="blue"))
    
    # Hiển thị hình ảnh liên quan
    images = answer_data.get("images", [])
    if images:
        img_table = Table(show_header=True, box=box.SIMPLE)
        img_table.add_column("Thứ tự", style="dim")
        img_table.add_column("Tên file", style="green")
        img_table.add_column("Nguồn", style="blue")
        img_table.add_column("Trang", style="cyan")
        
        for i, img in enumerate(images, 1):
            img_table.add_row(
                str(i),
                img.get("filename", ""),
                img.get("source", ""),
                str(img.get("page", "")),
            )
        
        console.print(Panel(img_table, title=f"Hình ảnh liên quan ({len(images)})", border_style="yellow"))
    
    # Hiển thị nguồn tham khảo
    sources = answer_data.get("sources", [])
    if sources:
        source_table = Table(show_header=True, box=box.SIMPLE)
        source_table.add_column("Thứ tự", style="dim")
        source_table.add_column("Tài liệu", style="green")
        source_table.add_column("Trang", style="cyan")
        
        for i, source in enumerate(sources, 1):
            source_table.add_row(
                str(i),
                source.get("source", ""),
                str(source.get("page", "")),
            )
        
        console.print(Panel(source_table, title="Nguồn tham khảo", border_style="magenta"))


# Hàm hiển thị giao diện đánh giá
def display_feedback_form(question, answer):
    """Hiển thị giao diện đánh giá dưới dạng biểu mẫu"""
    from rich.console import Console
    from rich.prompt import IntPrompt, Prompt, Confirm
    
    console = Console()
    console.print("\n[bold cyan]♦ Đánh giá câu trả lời ♦[/bold cyan]")
    
    # Điểm đánh giá
    rating = IntPrompt.ask("[yellow]Đánh giá câu trả lời (1-5)[/yellow]", default=3, choices=["1", "2", "3", "4", "5"])
    
    # Các tiêu chí khác
    correct = Confirm.ask("[yellow]Câu trả lời có chính xác không?[/yellow]", default=True)
    helpful = Confirm.ask("[yellow]Câu trả lời có hữu ích không?[/yellow]", default=True)
    missing_info = Confirm.ask("[yellow]Câu trả lời có thiếu thông tin quan trọng không?[/yellow]", default=False)
    
    # Danh mục
    categories = []
    add_category = Confirm.ask("[yellow]Bạn có muốn thêm danh mục cho câu hỏi này không?[/yellow]", default=False)
    if add_category:
        category = Prompt.ask("[yellow]Nhập danh mục[/yellow]")
        categories.append(category)
    
    # Nhận xét
    comment = ""
    add_comment = Confirm.ask("[yellow]Bạn có muốn thêm nhận xét không?[/yellow]", default=False)
    if add_comment:
        comment = Prompt.ask("[yellow]Nhận xét của bạn[/yellow]")
    
    # Đề xuất câu trả lời tốt hơn
    suggested_answer = ""
    suggest_better = Confirm.ask("[yellow]Bạn có muốn đề xuất câu trả lời tốt hơn không?[/yellow]", default=False)
    if suggest_better:
        suggested_answer = Prompt.ask("[yellow]Đề xuất câu trả lời của bạn[/yellow]")
    
    # Tạo dữ liệu đánh giá
    feedback_data = {
        "rating": rating,
        "correct": correct,
        "helpful": helpful,
        "missing_info": missing_info,
        "categories": categories,
        "comment": comment,
        "suggested_answer": suggested_answer,
        "timestamp": datetime.now().isoformat()
    }
    
    return feedback_data


## Hàm main để chạy chương trình
def main():
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import box
    
    console = Console()
    
    # Hiển thị banner
    console.print(Panel.fit(
        "[bold yellow]Hệ thống RAG Giáo dục Nâng cao[/bold yellow]\n"
        "[cyan]Hỗ trợ lịch sử hội thoại, trích xuất hình ảnh, đánh giá chất lượng và học từ phản hồi[/cyan]",
        border_style="green"
    ))
    
    # Đảm bảo thư mục books tồn tại
    if not os.path.exists("books/"):
        os.makedirs("books/")
        console.print("[bold red]Đã tạo thư mục books/. Vui lòng thêm tài liệu PDF vào thư mục này và chạy lại chương trình.[/bold red]")
        return
    
    # Kiểm tra số lượng file PDF trong thư mục books/
    pdf_files = [f for f in os.listdir("books/") if f.endswith('.pdf')]
    if not pdf_files:
        console.print("[bold red]Không tìm thấy file PDF nào trong thư mục books/. Vui lòng thêm tài liệu và chạy lại chương trình.[/bold red]")
        return
    
    # Hiển thị thông báo đang khởi tạo
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Đang khởi tạo hệ thống... Vui lòng đợi...[/bold green]")
    ) as progress:
        progress_task = progress.add_task("", total=None)
        
        # Khởi tạo EducationalRAG
        rag = EducationalRAG(pdf_dir="books/")
        
        # Cập nhật cơ sở kiến thức
        rag.update_knowledge_base()
    
    # Tạo phiên hội thoại mới
    session_id = rag.create_new_conversation()
    console.print(f"[green]Đã tạo phiên hội thoại mới với ID: {session_id}[/green]")
    
    # In thống kê
    stats = rag.get_statistics()
    
    stats_table = Table(show_header=False, box=box.ROUNDED)
    stats_table.add_column("Thông số", style="cyan")
    stats_table.add_column("Giá trị", style="yellow")
    
    stats_table.add_row("Tổng số tài liệu", str(stats['total_documents']))
    stats_table.add_row("Tổng số trang", str(stats['total_pages']))
    stats_table.add_row("Tổng số hình ảnh", str(stats.get('total_images', 0)))
    stats_table.add_row("Cập nhật lần cuối", stats.get('last_update', 'Chưa cập nhật'))
    
    console.print(Panel(stats_table, title="Thống kê hệ thống", border_style="blue"))
    
    # Danh sách câu hỏi mẫu
    questions = [
        "Phân biệt giữa thời tiết và khí hậu. Cho ví dụ minh họa.",
        "Mô tả đặc điểm của các đới khí hậu chính trên Trái Đất (nhiệt đới, ôn đới, hàn đới).",
        "Hiện tượng El Nino là gì và tác động của nó đến khí hậu Việt Nam?",
        "Trình bày các đặc điểm của chế độ mưa ở Việt Nam. Minh họa bằng biểu đồ nếu có.",
        "Biến đổi khí hậu ảnh hưởng như thế nào đến nông nghiệp Việt Nam?"
    ]
    
    # Chạy interactive mode để người dùng có thể hỏi câu hỏi
    console.print("\n[bold cyan]♦ Trợ lý học tập thông minh ♦[/bold cyan]")
    console.print("[dim]Nhập 'exit' để thoát, 'update' để cập nhật cơ sở kiến thức, 'stats' để xem thống kê.[/dim]")
    console.print("[dim]Nhập 'sample' để xem danh sách câu hỏi mẫu, 'sessions' để quản lý phiên, 'feedback' để xem phản hồi.[/dim]\n")
    
    while True:
        question = Prompt.ask("\n[bold green]Nhập câu hỏi của bạn[/bold green]")
        
        if question.lower() == 'exit':
            break
        elif question.lower() == 'update':
            console.print("\n[bold yellow]Đang cập nhật cơ sở kiến thức...[/bold yellow]")
            # Xóa vector store hiện tại để buộc tạo lại
            if os.path.exists(f"{rag.vector_db_path}.faiss"):
                try:
                    os.remove(f"{rag.vector_db_path}.faiss")
                    os.remove(f"{rag.vector_db_path}.pkl")
                    console.print("[green]Đã xóa vector store cũ.[/green]")
                except Exception as e:
                    console.print(f"[red]Lỗi khi xóa vector store cũ: {e}[/red]")
            
            # Reset vector store trong đối tượng
            rag.vector_store = None
            
            # Cập nhật với force_reload
            with Progress(SpinnerColumn(), TextColumn("[bold green]Đang cập nhật...[/bold green]")) as progress:
                task = progress.add_task("", total=None)
                rag.update_knowledge_base(force_reload=True)
            
            console.print("[bold green]Đã cập nhật xong![/bold green]")
            continue
        elif question.lower() == 'stats':
            stats = rag.get_statistics()
            
            stats_table = Table(show_header=False, box=box.ROUNDED)
            stats_table.add_column("Thông số", style="cyan")
            stats_table.add_column("Giá trị", style="yellow")
            
            stats_table.add_row("Tổng số tài liệu", str(stats['total_documents']))
            stats_table.add_row("Tổng số trang", str(stats['total_pages']))
            stats_table.add_row("Tổng số hình ảnh", str(stats.get('total_images', 0)))
            stats_table.add_row("Số phiên hội thoại", str(stats.get('conversation_sessions', 0)))
            stats_table.add_row("Số phản hồi đánh giá", str(stats.get('feedback_count', 0)))
            stats_table.add_row("Cập nhật lần cuối", stats.get('last_update', 'Chưa cập nhật'))
            
            if stats.get('documents'):
                doc_table = Table(show_header=True, box=box.SIMPLE)
                doc_table.add_column("Tên tài liệu")
                doc_table.add_column("Số trang")
                
                for doc in stats['documents']:
                    doc_table.add_row(doc['filename'], str(doc['pages']))
                    
                console.print(Panel(Columns([stats_table, doc_table]), title="Thống kê hệ thống", border_style="blue"))
            else:
                console.print(Panel(stats_table, title="Thống kê hệ thống", border_style="blue"))
            
            continue
        elif question.lower() == 'sample':
            console.print("\n[bold cyan]Danh sách câu hỏi mẫu:[/bold cyan]")
            for i, q in enumerate(questions, 1):
                console.print(f"[yellow]{i}.[/yellow] {q}")
            
            try:
                choice = Prompt.ask("\nChọn số thứ tự câu hỏi (hoặc Enter để quay lại)", default="")
                if choice.strip():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(questions):
                        question = questions[choice_idx]
                        console.print(f"\n[green]Câu hỏi đã chọn:[/green] {question}")
                    else:
                        console.print("[red]Số thứ tự không hợp lệ.[/red]")
                        continue
                else:
                    continue
            except ValueError:
                console.print("[red]Vui lòng nhập số thứ tự hợp lệ.[/red]")
                continue
        elif question.lower() == 'sessions':
            sessions = rag.get_conversation_sessions()
            
            if not sessions:
                console.print("[yellow]Chưa có phiên hội thoại nào.[/yellow]")
                continue
                
            console.print("[bold cyan]Danh sách phiên hội thoại:[/bold cyan]")
            session_table = Table(box=box.SIMPLE)
            session_table.add_column("ID", style="dim")
            session_table.add_column("Thời gian tạo", style="green")
            session_table.add_column("Cập nhật cuối", style="blue")
            session_table.add_column("Số tin nhắn", style="yellow")
            
            for session in sessions:
                session_table.add_row(
                    session["session_id"],
                    datetime.fromisoformat(session["created_at"]).strftime("%d/%m/%Y %H:%M"),
                    datetime.fromisoformat(session["last_updated"]).strftime("%d/%m/%Y %H:%M"),
                    str(session["message_count"])
                )
            
            console.print(session_table)
            
            action = Prompt.ask(
                "Chọn hành động",
                choices=["new", "load", "delete", "back"],
                default="back"
            )
            
            if action == "new":
                session_id = rag.create_new_conversation()
                console.print(f"[green]Đã tạo phiên hội thoại mới với ID: {session_id}[/green]")
            elif action == "load":
                session_id = Prompt.ask("Nhập ID phiên muốn tải")
                if rag.load_conversation_session(session_id):
                    console.print(f"[green]Đã tải phiên hội thoại {session_id}[/green]")
                else:
                    console.print("[red]Không tìm thấy phiên hội thoại.[/red]")
            elif action == "delete":
                session_id = Prompt.ask("Nhập ID phiên muốn xóa")
                if rag.conversation_memory.delete_session(session_id):
                    console.print(f"[green]Đã xóa phiên hội thoại {session_id}[/green]")
                else:
                    console.print("[red]Không tìm thấy phiên hội thoại.[/red]")
            
            continue
        elif question.lower() == 'feedback':
            insights = rag.get_feedback_insights()
            
            if insights["total_feedbacks"] == 0:
                console.print("[yellow]Chưa có đánh giá nào.[/yellow]")
                continue
                
            insight_table = Table(show_header=False, box=box.ROUNDED)
            insight_table.add_column("Thống kê", style="cyan")
            insight_table.add_column("Giá trị", style="yellow")
            
            insight_table.add_row("Tổng số đánh giá", str(insights["total_feedbacks"]))
            insight_table.add_row("Điểm đánh giá trung bình", f"{insights['avg_rating']:.1f}/5.0")
            insight_table.add_row("Tỷ lệ đánh giá tốt", f"{insights['positive_rate']*100:.1f}%")
            
            console.print(Panel(insight_table, title="Thống kê đánh giá", border_style="magenta"))
            
            # Hiển thị từ khóa có vấn đề
            if insights.get("problematic_keywords"):
                keyword_table = Table(box=box.SIMPLE)
                keyword_table.add_column("Từ khóa", style="red")
                keyword_table.add_column("Điểm TB", style="yellow")
                keyword_table.add_column("Số lượt", style="blue")
                
                for kw in insights["problematic_keywords"]:
                    keyword_table.add_row(
                        kw["keyword"],
                        f"{kw['avg_rating']:.1f}",
                        str(kw["count"])
                    )
                
                console.print(Panel(keyword_table, title="Từ khóa cần cải thiện", border_style="red"))
            
            continue
        
        # Xử lý câu hỏi trống
        if not question.strip():
            console.print("[yellow]Vui lòng nhập câu hỏi.[/yellow]")
            continue
            
        # Hiển thị đang xử lý
        with Progress(SpinnerColumn(), TextColumn("[bold green]Đang tìm câu trả lời...[/bold green]")) as progress:
            task = progress.add_task("", total=None)
            try:
                answer_data = rag.answer_question(question, use_conversation_history=True, with_feedback=True)
            except Exception as e:
                console.print(f"\n[bold red]Đã xảy ra lỗi khi trả lời:[/bold red] {str(e)}")
                console.print("[yellow]Vui lòng thử lại hoặc nhập 'update' để cập nhật lại cơ sở kiến thức.[/yellow]")
                continue
        
        # Hiển thị kết quả
        display_rich_answer(answer_data)
        
        # Hỏi người dùng có muốn đánh giá không
        if Confirm.ask("[cyan]Bạn có muốn đánh giá câu trả lời không?[/cyan]", default=False):
            feedback_data = display_feedback_form(question, answer_data["answer"])
            rag.add_feedback(question, answer_data["answer"], feedback_data)
            console.print("[green]Cảm ơn bạn đã đánh giá![/green]")

if __name__ == "__main__":
    main()
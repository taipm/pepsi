import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

class ConversationMemoryManager:
    """Lớp quản lý bộ nhớ hội thoại và phiên"""
    
    def __init__(self, storage_dir="conversations"):
        """
        Khởi tạo quản lý bộ nhớ
        
        Args:
            storage_dir: Thư mục lưu trữ phiên trò chuyện
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Thư mục cho dữ liệu học tập
        self.learning_dir = os.path.join(storage_dir, "learning")
        os.makedirs(self.learning_dir, exist_ok=True)
    
    def create_new_session(self, name=None) -> str:
        """
        Tạo phiên trò chuyện mới
        
        Args:
            name: Tên phiên (tùy chọn)
            
        Returns:
            str: ID của phiên mới
        """
        session_id = str(uuid.uuid4())
        
        if name is None:
            name = f"Phiên {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        session_data = {
            "id": session_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "exchanges": [],
            "feedback": []
        }
        
        self._save_session(session_id, session_data)
        return session_id
    
    def _save_session(self, session_id: str, session_data: Dict):
        """
        Lưu dữ liệu phiên vào file
        
        Args:
            session_id: ID phiên
            session_data: Dữ liệu phiên
        """
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Kiểm tra xem phiên có tồn tại không
        
        Args:
            session_id: ID phiên
            
        Returns:
            bool: True nếu phiên tồn tại
        """
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        return os.path.isfile(session_path)
    
    def load_session(self, session_id: str) -> Dict:
        """
        Tải dữ liệu phiên
        
        Args:
            session_id: ID phiên
            
        Returns:
            Dict: Dữ liệu phiên
        """
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.isfile(session_path):
            raise FileNotFoundError(f"Không tìm thấy phiên {session_id}")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_sessions(self) -> List[Dict]:
        """
        Liệt kê tất cả các phiên đã lưu
        
        Returns:
            List[Dict]: Danh sách thông tin phiên
        """
        sessions = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    session_path = os.path.join(self.storage_dir, filename)
                    with open(session_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        sessions.append({
                            "id": session_data.get("id"),
                            "name": session_data.get("name"),
                            "created_at": session_data.get("created_at"),
                            "updated_at": session_data.get("updated_at"),
                            "exchanges_count": len(session_data.get("exchanges", [])),
                            "feedback_count": len(session_data.get("feedback", []))
                        })
                except Exception as e:
                    print(f"Lỗi khi đọc phiên {filename}: {e}")
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return sessions
    
    def save_exchange(self, session_id: str, question: str, answer: str, 
                     source_docs: List[Dict] = None, images: List[Dict] = None,
                     quality_score: float = None):
        """
        Lưu một lượt trao đổi vào phiên
        
        Args:
            session_id: ID phiên
            question: Câu hỏi
            answer: Câu trả lời
            source_docs: Tài liệu nguồn
            images: Danh sách hình ảnh (không lưu dữ liệu base64)
            quality_score: Điểm đánh giá chất lượng
        """
        if not self.session_exists(session_id):
            raise ValueError(f"Phiên {session_id} không tồn tại")
        
        # Tải phiên hiện tại
        session_data = self.load_session(session_id)
        
        # Chuẩn bị thông tin hình ảnh (không lưu dữ liệu base64 để tiết kiệm không gian)
        image_info = []
        if images:
            for img in images:
                image_info.append({
                    "filename": img.get("filename"),
                    "source": img.get("source"),
                    "page": img.get("page")
                })
        
        # Thêm trao đổi mới
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "source_docs": source_docs if source_docs else [],
            "images": image_info,
            "quality_score": quality_score
        }
        
        session_data["exchanges"].append(exchange)
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Lưu lại phiên
        self._save_session(session_id, session_data)
    
    def add_feedback(self, session_id: str, question: str, answer: str, 
                    feedback_type: str, feedback_text: str = None):
        """
        Thêm phản hồi cho một câu trả lời
        
        Args:
            session_id: ID phiên
            question: Câu hỏi
            answer: Câu trả lời
            feedback_type: Loại phản hồi ("positive", "negative", "neutral")
            feedback_text: Nội dung phản hồi chi tiết
        """
        if not self.session_exists(session_id):
            raise ValueError(f"Phiên {session_id} không tồn tại")
        
        # Tải phiên hiện tại
        session_data = self.load_session(session_id)
        
        # Thêm phản hồi mới
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text
        }
        
        session_data["feedback"].append(feedback)
        session_data["updated_at"] = datetime.now().isoformat()
        
        # Lưu lại phiên
        self._save_session(session_id, session_data)
    
    def save_learning(self, session_id: str, question: str, answer: str, 
                    feedback: str, improvement: str):
        """
        Lưu bài học từ phản hồi người dùng
        
        Args:
            session_id: ID phiên
            question: Câu hỏi
            answer: Câu trả lời
            feedback: Phản hồi của người dùng
            improvement: Phân tích cải thiện
        """
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "improvement": improvement
        }
        
        # Tạo ID cho bài học
        learning_id = str(uuid.uuid4())
        
        # Lưu bài học vào file
        learning_path = os.path.join(self.learning_dir, f"{learning_id}.json")
        with open(learning_path, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, ensure_ascii=False, indent=2)
    
    def get_learning_insights(self) -> List[Dict]:
        """
        Lấy insight từ các bài học đã lưu
        
        Returns:
            List[Dict]: Danh sách các insight học hỏi
        """
        insights = []
        
        for filename in os.listdir(self.learning_dir):
            if filename.endswith('.json'):
                try:
                    learning_path = os.path.join(self.learning_dir, filename)
                    with open(learning_path, 'r', encoding='utf-8') as f:
                        learning_data = json.load(f)
                        insights.append({
                            "timestamp": learning_data.get("timestamp"),
                            "question": learning_data.get("question"),
                            "feedback": learning_data.get("feedback"),
                            "improvement": learning_data.get("improvement")
                        })
                except Exception as e:
                    print(f"Lỗi khi đọc bài học {filename}: {e}")
        
        # Sắp xếp theo thời gian (mới nhất trước)
        insights.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return insights
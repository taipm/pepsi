import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
from langchain.evaluation import load_evaluator

class QualityEvaluator:
    """Lớp đánh giá chất lượng câu trả lời"""
    
    def __init__(self, llm):
        """
        Khởi tạo đánh giá viên chất lượng
        
        Args:
            llm: Mô hình ngôn ngữ để sử dụng cho đánh giá
        """
        self.llm = llm
        self.evaluation_history = []
    
    def evaluate_answer(self, question: str, answer: str, source_docs: List[Dict]) -> float:
        """
        Đánh giá chất lượng câu trả lời
        
        Args:
            question: Câu hỏi
            answer: Câu trả lời
            source_docs: Danh sách tài liệu nguồn
            
        Returns:
            float: Điểm chất lượng từ 0 đến 1
        """
        try:
            # Tạo prompt đánh giá
            evaluation_prompt = f"""
            Hãy đánh giá chất lượng của câu trả lời dựa trên các tiêu chí sau:
            1. Độ chính xác: Câu trả lời có chính xác và phù hợp với câu hỏi không?
            2. Đầy đủ: Câu trả lời có đầy đủ thông tin cần thiết không?
            3. Rõ ràng: Câu trả lời có rõ ràng và dễ hiểu không?
            4. Trích dẫn nguồn: Câu trả lời có trích dẫn nguồn đầy đủ không?
            5. Liên quan: Câu trả lời có liên quan đến câu hỏi không?
            
            Câu hỏi: {question}
            
            Câu trả lời: {answer}
            
            Nguồn tham khảo: {', '.join([f"{doc.get('source', 'không rõ')} (trang {doc.get('page', 'không rõ')})" for doc in source_docs])}
            
            Điểm cho từng tiêu chí (thang điểm 1-10):
            """
            
            # Gọi LLM để đánh giá
            evaluation_result = self.llm.invoke(evaluation_prompt).content
            
            # Trích xuất điểm từ câu trả lời
            scores = self._extract_scores(evaluation_result)
            
            # Tính điểm trung bình
            if scores:
                avg_score = sum(scores) / len(scores)
                normalized_score = avg_score / 10.0  # Chuẩn hóa về 0-1
            else:
                normalized_score = 0.5  # Giá trị mặc định nếu không trích xuất được điểm
            
            # Lưu kết quả đánh giá
            evaluation_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "source_docs": source_docs,
                "evaluation_result": evaluation_result,
                "scores": scores,
                "normalized_score": normalized_score
            }
            
            self.evaluation_history.append(evaluation_entry)
            
            return normalized_score
            
        except Exception as e:
            print(f"Lỗi khi đánh giá câu trả lời: {e}")
            return 0.5  # Giá trị mặc định khi có lỗi
    
    def _extract_scores(self, evaluation_text: str) -> List[float]:
        """
        Trích xuất điểm từ kết quả đánh giá
        
        Args:
            evaluation_text: Kết quả đánh giá từ LLM
            
        Returns:
            List[float]: Danh sách các điểm
        """
        scores = []
        
        # Tìm tất cả các số từ 1-10 trong văn bản
        import re
        
        # Tìm các dòng có dạng "Tiêu chí X: Y" hoặc "X: Y" hoặc "X - Y" hoặc có số ở cuối dòng
        score_patterns = [
            r"(\d+)\/10",  # dạng 8/10
            r"(\d+)[/\.]\s*10",  # dạng 8/10 hoặc 8.10
            r"điểm\s*[:=]\s*(\d+)",  # dạng "điểm: 8"
            r"[:=]\s*(\d+)[/\.]?\d*\s*[\n$]",  # dạng ": 8" hoặc "= 8"
            r"[:=]\s*(\d+)[/\.]?\d*\s*[/của]?\s*10",  # dạng ": 8/10"
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, evaluation_text, re.IGNORECASE)
            for match in matches:
                try:
                    score = float(match)
                    if 0 <= score <= 10:
                        scores.append(score)
                except:
                    continue
        
        # Nếu không tìm thấy đủ điểm, tìm tất cả các số từ 1-10
        if len(scores) < 3:
            all_numbers = re.findall(r'\b(\d+)[/\.]?\d*\b', evaluation_text)
            for num in all_numbers:
                try:
                    score = float(num)
                    if 1 <= score <= 10 and score not in scores:
                        scores.append(score)
                except:
                    continue
        
        return scores
    
    def get_evaluation_history(self):
        """Lấy lịch sử đánh giá"""
        return self.evaluation_history


class ImageExtractor:
    """Lớp trích xuất hình ảnh từ tài liệu PDF"""
    
    def __init__(self, min_size=100):
        """
        Khởi tạo trình trích xuất hình ảnh
        
        Args:
            min_size: Kích thước tối thiểu của hình ảnh (chiều rộng hoặc chiều cao) để lưu
        """
        self.min_size = min_size
    
    def extract_images_from_pdf(self, pdf_path: str) -> Dict[int, List[Image.Image]]:
        """
        Trích xuất hình ảnh từ file PDF
        
        Args:
            pdf_path: Đường dẫn đến file PDF
            
        Returns:
            Dict: Từ điển với key là số trang, value là danh sách các đối tượng hình ảnh
        """
        images_by_page = {}
        
        try:
            # Mở file PDF
            pdf_document = fitz.open(pdf_path)
            
            # Lặp qua từng trang
            for page_num, page in enumerate(pdf_document):
                image_list = []
                
                # Lấy danh sách hình ảnh trên trang
                image_list = self._extract_images_from_page(page)
                
                if image_list:
                    images_by_page[page_num + 1] = image_list
            
            return images_by_page
        
        except Exception as e:
            print(f"Lỗi khi trích xuất hình ảnh từ {pdf_path}: {e}")
            return {}
    
    def _extract_images_from_page(self, page) -> List[Image.Image]:
        """
        Trích xuất tất cả hình ảnh từ một trang PDF
        
        Args:
            page: Đối tượng trang từ PyMuPDF
            
        Returns:
            List: Danh sách các đối tượng PIL Image
        """
        images = []
        
        # Lấy danh sách hình ảnh
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]  # Lấy tham chiếu hình ảnh
                base_image = page.parent.extract_image(xref)
                
                if not base_image:
                    continue
                
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Chuyển đổi thành đối tượng PIL Image
                image = Image.open(BytesIO(image_bytes))
                
                # Chỉ lưu các hình ảnh đủ lớn
                if image.width >= self.min_size and image.height >= self.min_size:
                    images.append(image)
            
            except Exception as e:
                print(f"Lỗi khi xử lý hình ảnh {img_index} trên trang {page.number + 1}: {e}")
                continue
                
        # Phương pháp trích xuất bổ sung bằng cách render trang
        # Hữu ích cho các hình ảnh phức tạp hoặc hình vẽ vector
        try:
            # Vẽ trang với độ phân giải cao
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Thêm hình ảnh toàn trang vào danh sách nếu không có hình ảnh nào khác
            if not images and img.width >= self.min_size and img.height >= self.min_size:
                images.append(img)
        except Exception as e:
            print(f"Lỗi khi render trang {page.number + 1}: {e}")
        
        return images
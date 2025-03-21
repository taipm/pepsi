import os
import json
import time
import hashlib
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

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
        
    def _load_or_create_metadata(self):
        """Tải metadata hoặc tạo mới nếu chưa tồn tại"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"files": {}, "last_update": None}
    
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
    
    def create_qa_chain(self):
        """Tạo chain QA với tham chiếu nguồn"""
        if self.vector_store is None:
            raise ValueError("Vector store chưa được khởi tạo. Hãy cập nhật cơ sở kiến thức trước.")
        
        # Thiết lập retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Tùy chỉnh prompt với yêu cầu trích dẫn nguồn
        template = """
        Bạn là trợ lý học tập thông minh dựa trên tài liệu sách giáo khoa.
        
        ### Câu hỏi:
        {question}
        
        ### Thông tin từ tài liệu:
        {context}
        
        ### Hướng dẫn:
        1. Trả lời dựa trên thông tin từ tài liệu được cung cấp.
        2. Nếu không tìm thấy thông tin, hãy thừa nhận điều đó thay vì đưa ra thông tin không chính xác.
        3. Phân tích các dữ liệu từ nhiều nguồn khác nhau để đưa ra câu trả lời toàn diện.
        4. Sắp xếp câu trả lời theo cấu trúc rõ ràng và dễ hiểu.
        5. Kết thúc câu trả lời bằng cách liệt kê các nguồn tài liệu tham khảo với format:
           [Nguồn tham khảo: <tên tài liệu>, trang <số trang>]
        
        ### Trả lời:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )
        
        # Tạo chain QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def answer_question(self, question):
        """Trả lời câu hỏi với tham chiếu nguồn"""
        if self.vector_store is None:
            print("Chưa có tài liệu nào. Đang cập nhật cơ sở kiến thức...")
            success = self.update_knowledge_base(force_reload=True)
            
            if not success or self.vector_store is None:
                print("Không thể tạo vector store. Đang thử tạo lại từ đầu...")
                # Xóa metadata để buộc xử lý lại toàn bộ tài liệu
                self.metadata = {"files": {}, "last_update": None}
                self._save_metadata()
                success = self.update_knowledge_base(force_reload=True)
                
                if not success or self.vector_store is None:
                    return "Không thể tạo vector store. Vui lòng kiểm tra lại tài liệu và thư mục."
        
        # Tạo chain QA
        try:
            qa_chain = self.create_qa_chain()
            
            # Đo thời gian truy vấn
            start_time = time.time()
            
            # Thực hiện truy vấn
            result = qa_chain({"query": question})
            
            # Xử lý kết quả để đảm bảo có tham chiếu
            answer = result["result"]
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
            
            query_time = time.time() - start_time
            print(f"Thời gian truy vấn: {query_time:.2f} giây")
            
            return answer
        except Exception as e:
            print(f"Lỗi khi trả lời câu hỏi: {str(e)}")
            return f"Đã xảy ra lỗi khi tìm câu trả lời: {str(e)}. Vui lòng thử lại hoặc cập nhật cơ sở kiến thức."
    
    def get_statistics(self):
        """Trả về thống kê về dữ liệu đã xử lý"""
        stats = {
            "total_documents": len(self.metadata["files"]),
            "total_pages": sum(file_info.get("total_pages", 0) for file_info in self.metadata["files"].values()),
            "last_update": datetime.fromtimestamp(self.metadata["last_update"]).strftime('%Y-%m-%d %H:%M:%S') if self.metadata["last_update"] else None,
            "documents": [{"filename": f, "pages": info.get("total_pages", 0)} for f, info in self.metadata["files"].items()]
        }
        return stats

## Hàm main để chạy chương trình
def main():
    # Đảm bảo thư mục books tồn tại
    if not os.path.exists("books/"):
        os.makedirs("books/")
        print("Đã tạo thư mục books/. Vui lòng thêm tài liệu PDF vào thư mục này và chạy lại chương trình.")
        return
    
    # Kiểm tra số lượng file PDF trong thư mục books/
    pdf_files = [f for f in os.listdir("books/") if f.endswith('.pdf')]
    if not pdf_files:
        print("Không tìm thấy file PDF nào trong thư mục books/. Vui lòng thêm tài liệu và chạy lại chương trình.")
        return
    
    # Khởi tạo EducationalRAG
    rag = EducationalRAG(pdf_dir="books/")
    
    # Cập nhật cơ sở kiến thức
    rag.update_knowledge_base()
    
    # In thống kê
    stats = rag.get_statistics()
    print("\nThống kê:")
    print(f"Tổng số tài liệu: {stats['total_documents']}")
    print(f"Tổng số trang: {stats['total_pages']}")
    print(f"Cập nhật lần cuối: {stats['last_update']}\n")
    
    # Danh sách câu hỏi mẫu
    questions = [
        "Phân biệt giữa thời tiết và khí hậu. Cho ví dụ minh họa.",
        "Mô tả đặc điểm của các đới khí hậu chính trên Trái Đất (nhiệt đới, ôn đới, hàn đới)."
    ]
    
    # Chạy interactive mode để người dùng có thể hỏi câu hỏi
    print("=== Trợ lý học tập ===")
    print("Nhập 'exit' để thoát, 'update' để cập nhật cơ sở kiến thức, 'stats' để xem thống kê.")
    print("Nhập 'sample' để xem danh sách câu hỏi mẫu.\n")
    
    while True:
        question = input("\nNhập câu hỏi của bạn: ")
        
        if question.lower() == 'exit':
            break
        elif question.lower() == 'update':
            print("\nĐang cập nhật cơ sở kiến thức...")
            # Xóa vector store hiện tại để buộc tạo lại
            if os.path.exists(f"{rag.vector_db_path}.faiss"):
                try:
                    os.remove(f"{rag.vector_db_path}.faiss")
                    os.remove(f"{rag.vector_db_path}.pkl")
                    print("Đã xóa vector store cũ.")
                except Exception as e:
                    print(f"Lỗi khi xóa vector store cũ: {e}")
            
            # Reset vector store trong đối tượng
            rag.vector_store = None
            
            # Cập nhật với force_reload
            rag.update_knowledge_base(force_reload=True)
            print("Đã cập nhật xong!")
            continue
        elif question.lower() == 'stats':
            stats = rag.get_statistics()
            print("\nThống kê hiện tại:")
            print(f"Tổng số tài liệu: {stats['total_documents']}")
            print(f"Tổng số trang: {stats['total_pages']}")
            print(f"Cập nhật lần cuối: {stats['last_update']}")
            if stats['documents']:
                print("\nDanh sách tài liệu:")
                for doc in stats['documents']:
                    print(f"- {doc['filename']}: {doc['pages']} trang")
            continue
        elif question.lower() == 'sample':
            print("\nDanh sách câu hỏi mẫu:")
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")
            
            try:
                choice = input("\nChọn số thứ tự câu hỏi (hoặc Enter để quay lại): ")
                if choice.strip():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(questions):
                        question = questions[choice_idx]
                        print(f"\nCâu hỏi đã chọn: {question}")
                    else:
                        print("Số thứ tự không hợp lệ.")
                        continue
                else:
                    continue
            except ValueError:
                print("Vui lòng nhập số thứ tự hợp lệ.")
                continue
        
        # Xử lý câu hỏi trống
        if not question.strip():
            print("Vui lòng nhập câu hỏi.")
            continue
            
        print("\nĐang tìm câu trả lời...")
        try:
            answer = rag.answer_question(question)
            print(f"\nCâu trả lời: {answer}")
        except Exception as e:
            print(f"\nĐã xảy ra lỗi khi trả lời: {str(e)}")
            print("Vui lòng thử lại hoặc nhập 'update' để cập nhật lại cơ sở kiến thức.")

if __name__ == "__main__":
    main()
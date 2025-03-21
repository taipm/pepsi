# Import các thư viện cần thiết
import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# Định nghĩa các hằng số và cấu hình
PDF_DIR = "books/"  # Thư mục chứa các file PDF sách giáo khoa
PROCESSED_PDFS_FILE = "processed_pdfs.txt"  # File lưu danh sách các PDF đã xử lý
CHUNK_SIZE = 1000  # Kích thước mỗi đoạn văn bản
CHUNK_OVERLAP = 200  # Độ chồng lấp giữa các đoạn
EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"  # Mô hình embedding tiếng Việt
FAISS_INDEX_PATH = "faiss_index"  # Đường dẫn lưu vectorstore FAISS
LLM_MODEL = "mrjacktung/phogpt-4b-chat-gguf"  # Mô hình ngôn ngữ PhoGPT
K_RETRIEVAL = 5  # Số lượng tài liệu trả về khi tìm kiếm

# Bước 1: Kiểm tra và cập nhật các file PDF mới
def load_processed_pdfs():
    """Đọc danh sách các file PDF đã xử lý từ file."""
    if os.path.exists(PROCESSED_PDFS_FILE):
        with open(PROCESSED_PDFS_FILE, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def detect_new_pdfs():
    """Xác định các file PDF mới chưa được xử lý."""
    processed_pdfs = load_processed_pdfs()
    all_pdfs = {f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')}
    return all_pdfs - processed_pdfs

def process_new_pdfs(new_pdfs):
    """Tải và xử lý các file PDF mới."""
    new_documents = []
    for pdf in new_pdfs:
        file_path = os.path.join(PDF_DIR, pdf)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
            new_documents.extend(docs)
            print(f"Đã tải {len(docs)} trang từ {pdf}")
        except Exception as e:
            print(f"Lỗi khi xử lý {pdf}: {e}")
    return new_documents

# Bước 2: Chia nhỏ văn bản thành các đoạn (chunks)
def split_documents(documents):
    """Chia nhỏ các tài liệu thành các đoạn văn bản."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents) if documents else []

# Bước 3: Tạo hoặc cập nhật vectorstore FAISS
def initialize_vectorstore():
    """Khởi tạo hoặc tải vectorstore FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(FAISS_INDEX_PATH):
        # Tải vectorstore đã có từ file
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Đã tải vectorstore từ file.")
    else:
        # Tạo vectorstore mới với HNSW index
        d = 768  # Kích thước embedding của mô hình (có thể thay đổi tùy mô hình)
        hnsw_index = faiss.IndexHNSWFlat(d, 32)  # Khởi tạo với d và M
        hnsw_index.hnsw.efConstruction = 200  # Thiết lập efConstruction
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=hnsw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}  # Cung cấp dictionary rỗng
        )
        print("Đã tạo vectorstore mới.")
    return vectorstore

def update_vectorstore(vectorstore, split_docs, new_pdfs):
    """Cập nhật vectorstore với các tài liệu mới và lưu lại."""
    if split_docs:
        vectorstore.add_documents(split_docs)
        vectorstore.save_local(FAISS_INDEX_PATH)
        with open(PROCESSED_PDFS_FILE, "a", encoding="utf-8") as f:
            for pdf in new_pdfs:
                f.write(pdf + "\n")
        print(f"Đã cập nhật vectorstore với {len(split_docs)} đoạn tài liệu mới.")

# Bước 4: Tạo pipeline RAG
def create_rag_pipeline(vectorstore):
    """Tạo pipeline RAG với retriever và mô hình ngôn ngữ."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Bước 5: Đặt câu hỏi và nhận câu trả lời
def ask_questions(qa, questions):
    """Đặt câu hỏi và hiển thị câu trả lời cùng tài liệu tham chiếu."""
    for question in questions:
        result = qa.invoke({"query": question})
        answer = result["result"]
        sources = result["source_documents"]
        
        # Trích xuất thông tin tài liệu tham chiếu
        references = []
        for doc in sources:
            pdf_path = doc.metadata.get("source", "Không xác định")
            page_number = doc.metadata.get("page", 0) + 1
            references.append(f"{pdf_path} - Trang {page_number}")
        
        # Hiển thị kết quả
        print(f"Câu hỏi: {question}")
        print(f"Câu trả lời: {answer}")
        print("Tài liệu tham chiếu:")
        for ref in references:
            print(f"- {ref}")
        print("=" * 50)

# Hàm chính để chạy chương trình
def main():
    # Kiểm tra và xử lý các file PDF mới
    new_pdfs = detect_new_pdfs()
    new_documents = process_new_pdfs(new_pdfs)
    split_docs = split_documents(new_documents)
    
    # Khởi tạo hoặc tải vectorstore
    vectorstore = initialize_vectorstore()
    
    # Cập nhật vectorstore nếu có tài liệu mới
    update_vectorstore(vectorstore, split_docs, new_pdfs)
    
    # Tạo pipeline RAG
    qa = create_rag_pipeline(vectorstore)
    
    # Danh sách câu hỏi mẫu
    questions = [
        "Phân biệt giữa thời tiết và khí hậu. Cho ví dụ minh họa.",
        "Mô tả đặc điểm của các đới khí hậu chính trên Trái Đất (nhiệt đới, ôn đới, hàn đới)."
    ]
    
    # Đặt câu hỏi và nhận câu trả lời
    ask_questions(qa, questions)

# Chạy chương trình
if __name__ == "__main__":
    main()
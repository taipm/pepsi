# Import các thư viện cần thiết
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Bước 1: Tải các file PDF từ thư mục chứa sách giáo khoa
pdf_dir = "books/"  # Thay bằng đường dẫn thực tế đến thư mục chứa file PDF
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

documents = []
for pdf in pdf_files:
    file_path = os.path.join(pdf_dir, pdf)
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    documents.extend(docs)
    print(f"Đã tải {len(docs)} trang từ {pdf}")

# Kiểm tra nội dung của 200 ký tự đầu tiên từ tài liệu đầu tiên
if documents:
    print("Nội dung mẫu:", documents[0].page_content[:200])

# Bước 2: Chia nhỏ văn bản thành các đoạn (chunk)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

# Bước 3: Tạo vector embedding và lưu trữ vào FAISS
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
vectorstore = FAISS.from_documents(split_documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Bước 5: Cấu hình mô hình PhoGPT-4B-Chat qua Ollama
llm = ChatOllama(model="mrjacktung/phogpt-4b-chat-gguf", temperature=0)

# Bước 7: Tích hợp retriever và generator vào pipeline RAG
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#TEST: CÁC CÂU HỎI ĐỊA LÝ DỰA TRÊN TÀI LIỆU
questions = [
    'Phân biệt giữa thời tiết và khí hậu. Cho ví dụ minh họa.',
    'Mô tả đặc điểm của các đới khí hậu chính trên Trái Đất (nhiệt đới, ôn đới, hàn đới).'
]
for question in questions:
    answer = qa.run(question)
    print(f"Câu hỏi: {question}\nCâu trả lời: {answer}\n")
    print(f'{"="*30}')

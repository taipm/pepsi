from langchain_community.document_loaders import PyPDFLoader
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

#STEP 5
# Thư mục chứa các file PDF
pdf_dir = "books/"

# Liệt kê tất cả các file PDF trong thư mục
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

# Danh sách để lưu trữ tất cả các document
documents = []

for pdf in pdf_files:
    file_path = os.path.join(pdf_dir, pdf)
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    documents.extend(docs)
    print(f"Đã tải {len(docs)} trang từ {pdf}")

# In ra nội dung của 200 ký tự đầu tiên của document đầu tiên để kiểm tra
if documents:
    print(documents[0].page_content[:200])


#STEP 6

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

#STEP 7-9

embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
vectorstore = FAISS.from_documents(split_documents, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#STEP 10

sample_question_math = "Giải thích định lý Pythagoras trong hình học."
relevant_docs_math = retriever.get_relevant_documents(sample_question_math)
for i, doc in enumerate(relevant_docs_math):
    print(f"Đoạn {i+1}: {doc.page_content[:200]}...")

#STEP 11
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mrjacktung/phogpt-4b-chat-gguf", temperature=0)


from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


question_math = "Giải thích định lý Pythagoras trong hình học."
answer_math = qa.run(question_math)
print("Câu trả lời:", answer_math)


from langchain_core.messages import HumanMessage, SystemMessage
context = "Định lý Pythagoras là một định lý quan trọng trong hình học, nêu rằng trong một tam giác vuông góc, bình phương cạnh huyền bằng tổng bình phương hai cạnh góc vuông."
question = "Giải thích định lý Pythagoras."
messages = [
    SystemMessage(content="Bạn là một trợ lý hữu ích trả lời các câu hỏi dựa trên ngữ cảnh."),
    HumanMessage(content=f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời:"),
]
answer = llm(messages)
print("Câu trả lời từ LLM:", answer.content)
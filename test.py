from langchain_core.messages import HumanMessage, SystemMessage
context = "Định lý Pythagoras là một định lý quan trọng trong hình học, nêu rằng trong một tam giác vuông góc, bình phương cạnh huyền bằng tổng bình phương hai cạnh góc vuông."
question = "Giải thích định lý Pythagoras."
messages = [
    SystemMessage(content="Bạn là một trợ lý hữu ích trả lời các câu hỏi dựa trên ngữ cảnh."),
    HumanMessage(content=f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời:"),
]
answer = llm(messages)
print("Câu trả lời từ LLM:", answer.content)



question_math = "Giải thích định lý Pythagoras trong hình học."
answer_math = qa.run(question_math)
print("Câu trả lời cho câu hỏi Toán:", answer_math)

question_literature = "Tóm tắt nội dung truyện 'Tấm Cám'."
answer_literature = qa.run(question_literature)
print("Câu trả lời cho câu hỏi Văn:", answer_literature)

question_history = "Nêu tên một sự kiện lịch sử quan trọng của Việt Nam thế kỷ 20."
answer_history = qa.run(question_history)
print("Câu trả lời cho câu hỏi Lịch sử:", answer_history)
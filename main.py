import os
import json
import time
import base64
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint
from PIL import Image
import io

from educational_rag import EducationalRAG

console = Console()

def display_image(image_data):
    """Hiển thị hình ảnh từ dữ liệu base64 (đối với terminal hỗ trợ)"""
    try:
        # Giải mã dữ liệu base64
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Cố gắng hiển thị trong terminal (một số terminal hỗ trợ)
        console.print("[yellow]Hình ảnh được tìm thấy nhưng không thể hiển thị trực tiếp trong terminal.[/yellow]")
        console.print(f"[green]Kích thước: {img.width}x{img.height}[/green]")
        
        # Lưu hình ảnh tạm thời
        temp_img_path = "temp_image.png"
        img.save(temp_img_path)
        console.print(f"[blue]Đã lưu hình ảnh tại: {temp_img_path}[/blue]")
    except Exception as e:
        console.print(f"[red]Lỗi khi hiển thị hình ảnh: {e}[/red]")

def display_answer(result):
    """Hiển thị câu trả lời với định dạng đẹp"""
    console.print("\n[bold green]Câu trả lời:[/bold green]")
    
    # Hiển thị câu trả lời dưới dạng markdown
    md = Markdown(result["answer"])
    console.print(Panel(md, border_style="green"))
    
    # Hiển thị điểm đánh giá chất lượng
    quality = result.get("quality_score", 0)
    quality_color = "green" if quality >= 0.7 else "yellow" if quality >= 0.4 else "red"
    console.print(f"[bold {quality_color}]Chất lượng câu trả lời: {quality:.2f}/1.0[/bold {quality_color}]")
    
    # Hiển thị hình ảnh nếu có
    images = result.get("images", [])
    if images:
        console.print(f"[bold cyan]Tìm thấy {len(images)} hình ảnh liên quan:[/bold cyan]")
        for i, img in enumerate(images):
            console.print(f"[cyan]Hình ảnh {i+1}: {img.get('filename')} (Trang {img.get('page')}, {img.get('source')})[/cyan]")
            
            # Hiển thị hình ảnh nếu có dữ liệu
            if "data" in img:
                display_image(img["data"])

def collect_feedback(rag, question, answer):
    """Thu thập phản hồi từ người dùng"""
    feedback_options = {
        "p": ("positive", "Tốt"),
        "n": ("negative", "Cần cải thiện"),
        "s": ("neutral", "Trung lập")
    }
    
    console.print("\n[bold yellow]Đánh giá câu trả lời:[/bold yellow]")
    for key, (_, desc) in feedback_options.items():
        console.print(f"[yellow]{key}[/yellow]: {desc}")
    
    choice = Prompt.ask("Lựa chọn của bạn", choices=list(feedback_options.keys()) + [""], default="")
    
    if choice and choice in feedback_options:
        feedback_type, _ = feedback_options[choice]
        
        if feedback_type == "negative":
            feedback_text = Prompt.ask("Bạn có thể cho biết cần cải thiện điều gì", default="")
        else:
            feedback_text = None
        
        # Xử lý phản hồi
        rag.process_feedback(question, answer, feedback_type, feedback_text)
        
        if feedback_type == "positive":
            console.print("[green]Cảm ơn phản hồi tích cực của bạn![/green]")
        elif feedback_type == "negative":
            console.print("[yellow]Cảm ơn phản hồi của bạn. Chúng tôi sẽ cải thiện câu trả lời trong tương lai.[/yellow]")
        else:
            console.print("[blue]Cảm ơn phản hồi của bạn.[/blue]")

def display_sessions(sessions):
    """Hiển thị danh sách phiên trò chuyện"""
    if not sessions:
        console.print("[yellow]Chưa có phiên trò chuyện nào.[/yellow]")
        return
    
    table = Table(title="Danh sách phiên trò chuyện")
    table.add_column("STT", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Tên", style="green")
    table.add_column("Ngày tạo", style="yellow")
    table.add_column("Số lượt trao đổi", style="magenta")
    
    for i, session in enumerate(sessions, 1):
        try:
            created_at = datetime.fromisoformat(session.get("created_at", "")).strftime("%d/%m/%Y %H:%M")
        except:
            created_at = "Không rõ"
            
        table.add_row(
            str(i),
            session.get("id", "")[:8] + "...",
            session.get("name", "Không tên"),
            created_at,
            str(session.get("exchanges_count", 0))
        )
    
    console.print(table)

def display_learning_insights(insights):
    """Hiển thị các insight học hỏi"""
    if not insights:
        console.print("[yellow]Chưa có bài học nào được lưu.[/yellow]")
        return
    
    console.print("[bold cyan]Những bài học từ phản hồi người dùng:[/bold cyan]")
    
    for i, insight in enumerate(insights[:5], 1):  # Chỉ hiển thị 5 bài học gần nhất
        try:
            timestamp = datetime.fromisoformat(insight.get("timestamp", "")).strftime("%d/%m/%Y %H:%M")
        except:
            timestamp = "Không rõ"
            
        console.print(f"\n[bold cyan]{i}. Bài học ({timestamp}):[/bold cyan]")
        console.print(f"[yellow]Câu hỏi:[/yellow] {insight.get('question', '')}")
        console.print(f"[yellow]Phản hồi người dùng:[/yellow] {insight.get('feedback', '')}")
        console.print(Panel(insight.get('improvement', ''), title="Phân tích cải thiện", border_style="blue"))

def display_statistics(stats):
    """Hiển thị thống kê hệ thống"""
    console.print("[bold cyan]Thống kê hệ thống:[/bold cyan]")
    
    table = Table()
    table.add_column("Thông số", style="cyan")
    table.add_column("Giá trị", style="green")
    
    table.add_row("Tổng số tài liệu", str(stats.get("total_documents", 0)))
    table.add_row("Tổng số trang", str(stats.get("total_pages", 0)))
    table.add_row("Tổng số hình ảnh", str(stats.get("total_images", 0)))
    table.add_row("Tổng số phiên trò chuyện", str(stats.get("total_sessions", 0)))
    table.add_row("Cập nhật lần cuối", stats.get("last_update", "Chưa cập nhật"))
    
    # Hiển thị thông tin phiên hiện tại nếu có
    if "current_session" in stats:
        current = stats["current_session"]
        table.add_row("Phiên hiện tại", current.get("name", ""))
        table.add_row("Số lượt trao đổi", str(current.get("exchanges", 0)))
    
    console.print(table)
    
    # Hiển thị danh sách tài liệu
    if stats.get("documents"):
        doc_table = Table(title="Danh sách tài liệu")
        doc_table.add_column("Tên tài liệu", style="cyan")
        doc_table.add_column("Số trang", style="green")
        
        for doc in stats.get("documents", []):
            doc_table.add_row(doc.get("filename", ""), str(doc.get("pages", 0)))
        
        console.print(doc_table)

def display_help():
    """Hiển thị trợ giúp"""
    help_text = """
# Trợ lý học tập thông minh

## Các lệnh có sẵn:
- **exit**: Thoát chương trình
- **update**: Cập nhật cơ sở kiến thức
- **stats**: Xem thống kê hệ thống
- **help**: Hiển thị trợ giúp
- **sample**: Xem danh sách câu hỏi mẫu
- **session**: Quản lý phiên trò chuyện
- **learning**: Xem các bài học từ phản hồi

## Cách sử dụng:
1. Nhập câu hỏi của bạn về các tài liệu đã được tải
2. Đánh giá câu trả lời để giúp hệ thống học hỏi
3. Sử dụng lệnh **update** để cập nhật khi có tài liệu mới
    """
    
    console.print(Markdown(help_text))

def session_menu(rag):
    """Menu quản lý phiên trò chuyện"""
    while True:
        console.print("\n[bold cyan]Quản lý phiên trò chuyện:[/bold cyan]")
        console.print("1. Xem danh sách phiên")
        console.print("2. Tạo phiên mới")
        console.print("3. Tải phiên đã lưu")
        console.print("0. Quay lại")
        
        choice = Prompt.ask("Lựa chọn của bạn", choices=["0", "1", "2", "3"], default="0")
        
        if choice == "0":
            break
        elif choice == "1":
            sessions = rag.list_sessions()
            display_sessions(sessions)
        elif choice == "2":
            name = Prompt.ask("Nhập tên cho phiên mới", default=f"Phiên {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            session_id = rag.start_new_session(name)
            console.print(f"[green]Đã tạo phiên mới với ID: {session_id}[/green]")
        elif choice == "3":
            sessions = rag.list_sessions()
            display_sessions(sessions)
            
            if sessions:
                idx = Prompt.ask("Chọn STT phiên muốn tải", default="1")
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(sessions):
                        session_id = sessions[idx]["id"]
                        if rag.load_session(session_id):
                            console.print(f"[green]Đã tải phiên {sessions[idx]['name']}[/green]")
                        else:
                            console.print("[red]Không thể tải phiên.[/red]")
                    else:
                        console.print("[red]STT không hợp lệ.[/red]")
                except ValueError:
                    console.print("[red]Vui lòng nhập số.[/red]")

def main():
    # Đảm bảo thư mục books tồn tại
    if not os.path.exists("books/"):
        os.makedirs("books/")
        console.print("[yellow]Đã tạo thư mục books/. Vui lòng thêm tài liệu PDF vào thư mục này và chạy lại chương trình.[/yellow]")
        return
    
    # Kiểm tra số lượng file PDF trong thư mục books/
    pdf_files = [f for f in os.listdir("books/") if f.endswith('.pdf')]
    if not pdf_files:
        console.print("[yellow]Không tìm thấy file PDF nào trong thư mục books/. Vui lòng thêm tài liệu và chạy lại chương trình.[/yellow]")
        return
    
    # Hiển thị màn hình chào mừng
    console.print(Panel.fit(
        "[bold green]Trợ lý học tập thông minh[/bold green]\n"
        "[cyan]Phiên bản nâng cao với memory, hình ảnh, đánh giá chất lượng và học chủ động[/cyan]",
        border_style="green"
    ))
    
    # Khởi tạo EducationalRAG
    console.print("\n[yellow]Đang khởi tạo hệ thống...[/yellow]")
    rag = EducationalRAG(pdf_dir="books/")
    
    # Cập nhật cơ sở kiến thức
    console.print("[yellow]Đang cập nhật cơ sở kiến thức...[/yellow]")
    rag.update_knowledge_base()
    
    # Tạo phiên mặc định
    rag.start_new_session("Phiên mặc định")
    
    # In thống kê
    stats = rag.get_statistics()
    display_statistics(stats)
    
    # Danh sách câu hỏi mẫu
    questions = [
        "Phân biệt giữa thời tiết và khí hậu. Cho ví dụ minh họa.",
        "Mô tả đặc điểm của các đới khí hậu chính trên Trái Đất (nhiệt đới, ôn đới, hàn đới).",
        "Hãy giải thích hiệu ứng nhà kính và tác động của nó đến biến đổi khí hậu.",
        "Các loại đá magma là gì? Cho ví dụ về mỗi loại.",
        "Trình bày về chu trình nước trong tự nhiên."
    ]
    
    # Chạy interactive mode để người dùng có thể hỏi câu hỏi
    console.print("\n[bold green]=== Trợ lý học tập thông minh ===[/bold green]")
    console.print("Nhập [bold]exit[/bold] để thoát, [bold]update[/bold] để cập nhật cơ sở kiến thức, [bold]stats[/bold] để xem thống kê.")
    console.print("Nhập [bold]help[/bold] để xem trợ giúp, [bold]sample[/bold] để xem danh sách câu hỏi mẫu.")
    console.print("Nhập [bold]session[/bold] để quản lý phiên trò chuyện, [bold]learning[/bold] để xem các bài học.\n")
    
    while True:
        question = Prompt.ask("\nNhập câu hỏi của bạn", default="")
        
        if not question.strip():
            continue
        
        # Xử lý các lệnh đặc biệt
        if question.lower() == 'exit':
            if Confirm.ask("Bạn có chắc muốn thoát?"):
                break
        elif question.lower() == 'update':
            console.print("\n[yellow]Đang cập nhật cơ sở kiến thức...[/yellow]")
            # Xóa vector store hiện tại để buộc tạo lại
            if os.path.exists(f"{rag.vector_db_path}.faiss"):
                try:
                    os.remove(f"{rag.vector_db_path}.faiss")
                    os.remove(f"{rag.vector_db_path}.pkl")
                    console.print("[yellow]Đã xóa vector store cũ.[/yellow]")
                except Exception as e:
                    console.print(f"[red]Lỗi khi xóa vector store cũ: {e}[/red]")
            
            # Reset vector store trong đối tượng
            rag.vector_store = None
            
            # Cập nhật với force_reload
            rag.update_knowledge_base(force_reload=True)
            console.print("[green]Đã cập nhật xong![/green]")
            continue
        elif question.lower() == 'stats':
            stats = rag.get_statistics()
            display_statistics(stats)
            continue
        elif question.lower() == 'help':
            display_help()
            continue
        elif question.lower() == 'sample':
            console.print("\n[bold cyan]Danh sách câu hỏi mẫu:[/bold cyan]")
            for i, q in enumerate(questions, 1):
                console.print(f"{i}. [cyan]{q}[/cyan]")
            
            try:
                choice = Prompt.ask("\nChọn số thứ tự câu hỏi (hoặc Enter để quay lại)", default="")
                if choice.strip():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(questions):
                        question = questions[choice_idx]
                        console.print(f"\n[cyan]Câu hỏi đã chọn: {question}[/cyan]")
                    else:
                        console.print("[red]Số thứ tự không hợp lệ.[/red]")
                        continue
                else:
                    continue
            except ValueError:
                console.print("[red]Vui lòng nhập số thứ tự hợp lệ.[/red]")
                continue
        elif question.lower() == 'session':
            session_menu(rag)
            continue
        elif question.lower() == 'learning':
            insights = rag.get_learning_insights()
            display_learning_insights(insights)
            continue
            
        # Xử lý câu hỏi trống
        if not question.strip():
            console.print("[yellow]Vui lòng nhập câu hỏi.[/yellow]")
            continue
            
        console.print(f"\n[bold]Đang tìm câu trả lời cho: [cyan]{question}[/cyan][/bold]")
        with console.status("[bold green]Đang xử lý...[/bold green]"):
            try:
                start_time = time.time()
                result = rag.answer_question(question)
                query_time = time.time() - start_time
                
                console.print(f"[dim](Thời gian xử lý: {query_time:.2f} giây)[/dim]")
                display_answer(result)
                
                # Thu thập phản hồi
                collect_feedback(rag, question, result["answer"])
                
            except Exception as e:
                console.print(f"\n[bold red]Đã xảy ra lỗi khi trả lời: {str(e)}[/bold red]")
                console.print("[yellow]Vui lòng thử lại hoặc nhập 'update' để cập nhật lại cơ sở kiến thức.[/yellow]")

if __name__ == "__main__":
    main()
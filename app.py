from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import PyPDF2
import re
import tomli
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from flask_session import Session  # ✅ để lưu session người dùng

# ================== CẤU HÌNH & KHỞI TẠO ==================
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)
api_key = config["secrets"]["GEMINI_API_KEY"]

if not api_key:
    raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong biến môi trường!")

genai.configure(api_key=api_key)

GENERATION_MODEL = 'gemini-2.5-flash-lite'
EMBEDDING_MODEL = 'text-embedding-004'

app = Flask(__name__)
app.secret_key = "1234"  # ⚠️ cần có key để session hoạt động
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Biến toàn cục cho RAG
RAG_DATA = {
    "chunks": [],
    "embeddings": np.array([]),
    "is_ready": False
}

# ================== ĐỌC & CHIA CHUNKS ==================
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc PDF {pdf_path}: {e}")
    return text

def create_chunks_from_directory(directory='./static', chunk_size=400):
    all_chunks = []
    if not os.path.exists(directory):
        print(f"Thư mục {directory} không tồn tại.")
        return []

    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    print(f"🔍 Tìm thấy {len(pdf_files)} tệp PDF trong {directory}...")

    if not pdf_files:
        return ["Giới thiệu về các dự án lập trình sáng tạo cho học sinh tiểu học như Scratch, Teachable Machine, trò chơi mini."]

    for filename in pdf_files:
        pdf_path = os.path.join(directory, filename)
        content = extract_pdf_text(pdf_path)
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size].strip()
            if chunk:
                all_chunks.append(f"[Nguồn: {filename}] {chunk}")
    print(f"✅ Đã tạo tổng cộng {len(all_chunks)} đoạn văn (chunks).")
    return all_chunks

def embed_with_retry(texts, model_name, max_retries=5):
    all_embeddings = []
    for text in texts:
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=model_name, content=text)
                all_embeddings.append(result["embedding"])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Thử lại lần {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"💥 Thất bại sau {max_retries} lần: {e}")
                    raise
    return np.array(all_embeddings)

def initialize_rag_data():
    global RAG_DATA
    print("⏳ Đang khởi tạo dữ liệu RAG...")
    chunks = create_chunks_from_directory()
    if not chunks:
        print("Không có dữ liệu để nhúng.")
        return
    try:
        embeddings = embed_with_retry(chunks, EMBEDDING_MODEL)
        RAG_DATA.update({
            "chunks": chunks,
            "embeddings": embeddings,
            "is_ready": True
        })
        print("🎉 Khởi tạo RAG hoàn tất!")
    except Exception as e:
        print(f"❌ KHÔNG THỂ KHỞI TẠO RAG: {e}")
        RAG_DATA["is_ready"] = False

initialize_rag_data()

# ================== TRUY XUẤT NGỮ CẢNH ==================
def retrieve_context(query, top_k=3):
    if not RAG_DATA["is_ready"]:
        return "Không có tài liệu RAG nào được tải."
    try:
        query_vec = embed_with_retry([query], EMBEDDING_MODEL)[0].reshape(1, -1)
        sims = cosine_similarity(query_vec, RAG_DATA["embeddings"])[0]
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        return "\n\n---\n\n".join([RAG_DATA["chunks"][i] for i in top_idxs])
    except Exception as e:
        print(f"❌ Lỗi RAG: {e}")
        return "Lỗi khi tìm kiếm ngữ cảnh."

# ================== ĐỊNH DẠNG TRẢ LỜI ==================
def format_response(response):
    """Định dạng văn bản trả lời thân thiện cho học sinh."""

    # 1️⃣ In đậm phần **...**
    formatted = re.sub(r'\*\*(.*?)\*\*',
                       r'<strong font-weight:700;">\1</strong>',
                       response)

    # 2️⃣ In nghiêng phần *...* (nhưng KHÔNG áp dụng cho đầu dòng gạch đầu dòng)
    formatted = re.sub(r'(?<!\n)\*(?!\s)(.*?)(?<!\s)\*(?!\*)',
                       r'<em font-style:italic;">\1</em>',
                       formatted)

    # 3️⃣ Biến các dòng bắt đầu bằng "*" thành gạch đầu dòng
    formatted = re.sub(
        r'(?m)^\s*\*\s+(.*)',
        r'• <span style="line-height:1.6;">\1</span>',
        formatted
    )

    # 4️⃣ Thay xuống dòng thành <br> cho hiển thị web
    formatted = formatted.replace('\n', '<br>')

    # 6️⃣ Thêm khung cho phần trả lời
    return f"""
    <div style="
        background:#FAFAFA;
        border-left:6px solid #FFB300;
        padding:10px 15px;
        border-radius:8px;
        line-height:2;
        font-size:15px;
        color:#212121;
        font-family:'Segoe UI', sans-serif;">
        {formatted}
    </div>
    """

# ================== ROUTES ==================
# === DỰ ÁN MẪU ===

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/')
def index():
    rag_status = "✅ Đã tải tài liệu RAG thành công" if RAG_DATA["is_ready"] else "⚠️ Chưa tải được tài liệu RAG."
    session.clear()  # 🧹 Mỗi lần reload trang sẽ reset hội thoại
    return render_template('index.html', rag_status=rag_status)

#FORMAT TRẢ LỜI
highlight_terms = {
    
    "🟦 Chuyển động (Motion)":"#4C97FF",
    "di chuyển (10) bước": "#4C97FF",
    "xoay ↻ (15) độ": "#4C97FF",
    "xoay ↺ (15) độ": "#4C97FF",
    "đi tới (vị trí ngẫu nhiên / con trỏ chuột / nhân vật …)": "#4C97FF",
    "đi tới điểm x: (0) y: (0)": "#4C97FF",
    "lướt trong (1) giây tới (vị trí ngẫu nhiên / con trỏ chuột / nhân vật …)": "#4C97FF",
    "lướt trong (1) giây tới điểm x: (0) y: (0)": "#4C97FF",
    "đặt hướng bằng (90)": "#4C97FF",
    "hướng về phía đối tượng (con trỏ chuột / nhân vật …)": "#4C97FF",
    "thay đổi x một lượng (10)": "#4C97FF",
    "đặt x bằng (0)": "#4C97FF",
    "thay đổi y một lượng (10)": "#4C97FF",
    "đặt y bằng (0)": "#4C97FF",
    "bật lại nếu chạm cạnh": "#4C97FF",
    "đặt kiểu xoay (trái - phải / toàn vòng / không xoay)": "#4C97FF",
    "tọa độ x": "#4C97FF",
    "tọa độ y": "#4C97FF",
    "hướng": "#4C97FF",

    # 🟪 Hiển thị (Looks)
    'nói ("Xin chào!") trong (2) giây': "#9966FF",
    'nói ("Xin chào!")': "#9966FF",
    'nghĩ ("Hmm...") trong (2) giây': "#9966FF",
    'nghĩ ("Hmm...")': "#9966FF",
    "chuyển sang trang phục (costume2)": "#9966FF",
    "trang phục kế tiếp": "#9966FF",
    "đổi phông nền thành (backdrop1)": "#9966FF",
    "phông nền tiếp theo": "#9966FF",
    "đổi kích thước một lượng (10)": "#9966FF",
    "đặt kích thước thành (100)%": "#9966FF",
    "thay đổi hiệu ứng (màu / sáng / xoáy / mờ / ảo ảnh) một lượng (25)": "#9966FF",
    "đặt hiệu ứng (màu / sáng / xoáy / mờ / ảo ảnh) bằng (0)": "#9966FF",
    "bỏ các hiệu ứng đồ họa": "#9966FF",
    "hiện": "#9966FF",
    "ẩn": "#9966FF",
    "đi tới lớp phía (trên cùng / dưới cùng)": "#9966FF",
    "đi tới (1) lớp": "#9966FF",
    "trang phục (số)": "#9966FF",
    "phông nền (số)": "#9966FF",
    "kích thước": "#9966FF",

    # 🟣 Âm thanh (Sound)
    "🟣 Âm thanh (Sound)":"#CF63CF",
    "phát âm thanh (Robot) đến hết": "#CF63CF",
    "bắt đầu âm thanh (Robot)": "#CF63CF",
    "ngừng mọi âm thanh": "#CF63CF",
    "thay đổi hiệu ứng (cao độ / vang / ...) một lượng (10)": "#CF63CF",
    "đặt hiệu ứng (cao độ / vang / ...) bằng (100)": "#CF63CF",
    "xóa hiệu ứng âm thanh": "#CF63CF",
    "thay đổi âm lượng một lượng (-10)": "#CF63CF",
    "đặt âm lượng (100)%": "#CF63CF",
    "âm lượng": "#CF63CF",

    # 🟨 Sự kiện (Events)
    "🟨 Sự kiện (Events)":"#FFBF00",
    "khi bấm vào cờ xanh": "#FFBF00",
    "khi bấm phím (phím trắng / phím cách / mũi tên ...)": "#FFBF00",
    "khi bấm vào nhân vật này": "#FFBF00",
    "khi phông nền chuyển thành (backdrop1)": "#FFBF00",
    "khi (độ ồn > 10)": "#FFBF00",
    "khi nhận tin nhắn (1)": "#FFBF00",
    "phát tin nhắn (1)": "#FFBF00",
    "phát tin nhắn (1) và đợi": "#FFBF00",

    # 🟧 Điều khiển (Control)
    "🟧 Điều khiển (Control)":"#FFAB19",
    "đợi (1) giây": "#FFAB19",
    "lặp lại (10) lần": "#FFAB19",
    "liên tục": "#FFAB19",
    "nếu (điều kiện) thì": "#FFAB19",
    "nếu (điều kiện) thì... nếu không thì...": "#FFAB19",
    "đợi đến khi (điều kiện)": "#FFAB19",
    "lặp lại cho đến khi (điều kiện)": "#FFAB19",
    "dừng lại (tất cả / tập lệnh này / nhân vật khác)": "#FFAB19",
    "khi tôi bắt đầu là một bản sao": "#FFAB19",
    "tạo bản sao của (bản thân tôi / nhân vật khác)": "#FFAB19",
    "xóa bản sao này": "#FFAB19",

    # 🩵 Cảm biến (Sensing)
    "🩵 Cảm biến (Sensing)":"#5CB1D6",
    "đang chạm (chuột / màu / nhân vật)": "#5CB1D6",
    "màu () đang chạm ()": "#5CB1D6",
    "khoảng cách đến (chuột / nhân vật)": "#5CB1D6",
    'hỏi ("Tên của bạn là gì?") và đợi': "#5CB1D6",
    "trả lời": "#5CB1D6",
    "phím (phím trắng) được bấm?": "#5CB1D6",
    "chuột được nhấn?": "#5CB1D6",
    "tọa độ x con trỏ chuột": "#5CB1D6",
    "tọa độ y con trỏ chuột": "#5CB1D6",
    "đặt chế độ kéo (kéo thả được / không)": "#5CB1D6",
    "độ ồn": "#5CB1D6",
    "đồng hồ bấm giờ": "#5CB1D6",
    "đặt lại đồng hồ bấm giờ": "#5CB1D6",
    "phông nền # của Sân khấu": "#5CB1D6",
    "năm hiện tại": "#5CB1D6",
    "tên đăng nhập": "#5CB1D6",

    # 🟩 Các phép toán (Operators)
    "🟩 Các phép toán (Operators)":"#59C059",
    "+": "#59C059",
    "−": "#59C059",
    "×": "#59C059",
    "÷": "#59C059",
    "lấy ngẫu nhiên từ (1) đến (10)": "#59C059",
    ">": "#59C059",
    "<": "#59C059",
    "=": "#59C059",
    "và": "#59C059",
    "hoặc": "#59C059",
    "không phải": "#59C059",
    "kết hợp (chuỗi)": "#59C059",
    "ký tự thứ (1) của (chuỗi)": "#59C059",
    "độ dài của (chuỗi)": "#59C059",
    "chia lấy dư": "#59C059",
    "làm tròn": "#59C059",
    "giá trị tuyệt đối": "#59C059",

    # 🟠 Các biến số (Variables)
    "🟠 Các biến số (Variables)":"#FF8C1A",
    "tạo một biến": "#FF8C1A",
    "đặt (biến của tôi) thành (0)": "#FF8C1A",
    "thay đổi (biến của tôi) một lượng (1)": "#FF8C1A",
    "hiển thị biến (biến của tôi)": "#FF8C1A",
    "ẩn biến (biến của tôi)": "#FF8C1A",
    "tạo một danh sách": "#FF8C1A",

    # 🟥 Khối của tôi (My Blocks)
    "🟥 Khối của tôi (My Blocks)":"#FF6680",
    "tạo một khối mới": "#FF6680",
    "chạy khối đã tạo": "#FF6680"
}


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    # 📚 Lưu hội thoại vào session
    if "history" not in session:
        session["history"] = []
    session["history"].append(f"👧 Học sinh: {user_message}")

    # 🔍 Truy xuất ngữ cảnh RAG
    related_context = retrieve_context(user_message)
    recent_history = "\n".join(session["history"][-5:])  # chỉ giữ 5 câu gần nhất

    prompt = f"""
    "TÁC GIẢ DỰ ÁN: HỌC SINH THỰC HIỆN - Nguyễn Ngô Tường Nguyên, Đặng Thành Kim - GIÁO VIÊN HƯỚNG DẪN: Trần Quốc Giăng, không trả lời nếu không cần thiết"
    Bạn là một **Thầy/Cô giáo dạy lập trình sáng tạo** (Scratch, Teachable Machine) cho học sinh tiểu học, xưng là thầy và con.
    Hãy trả lời ngắn gọn, thân thiện, dễ hiểu, trình bày theo từng bước thực hiện.
    Format màu cho các thẻ lệnh giúp học sinh dễ dàng tìm kiếm, Ưu tiên trả lời đúng từ khóa và các thẻ lệnh khi đưa ra hướng dẫn:{highlight_terms} 
    Đối với các câu lệnh được sử dụng hoặc có các từ khóa bên trong, sinh ra thẻ span bọc với mẫu <span style = "line-height:1.6; background: ( màu dựa trên highlight_terms); color:white; font-weight:bold; padding:2px 4px; border-radius:4px;"> (các từ khóa trong highlight_terms)</span>
    Tương tự với từ khóa "Scratch":"#FFAB19" và "Teachable Machine":"#CF63CF".
    🎯 Tài liệu RAG:
    {related_context}
    🗣️ Lịch sử hội thoại gần đây:
    {recent_history}
    Đối với các câu hỏi không có từ "sản phẩm mẫu" hoặc đại loại muốn tham khảo sản phẩm mẫu trong tài liệu, trả lời dựa theo lịch sử hội thoại các câu hỏi của học sinh.
    Đối với các câu hỏi muốn tham khảo từ sản phẩm mẫu trong tài liệu RAG, chỉ học sinh cách làm theo câu hỏi, dễ hiểu nhất và trình bày ngắn gọn.
    🧠 Câu hỏi mới: {user_message}
    """

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        ai_text = response.text

        # 🧑‍🏫 Lưu trả lời AI vào session
        session["history"].append(f"🧑‍🏫 Thầy/Cô: {ai_text}")
        session.modified = True  # để Flask lưu lại session

        return jsonify({'response': format_response(ai_text)})

    except Exception as e:
        print(f"❌ Lỗi Gemini: {e}")
        return jsonify({'response': format_response("Thầy Gemini hơi mệt, con thử lại sau nhé!")})

# ================== CHẠY APP ==================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

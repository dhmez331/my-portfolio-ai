import os
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
from dotenv import load_dotenv

# المكتبات الأساسية لـ RAG
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

# --- إعدادات الإيميل ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

vector_store = None

def initialize_rag():
    """تهيئة ذاكرة الذكاء الاصطناعي وقراءة ملفات الـ PDF"""
    global vector_store
    
    # استخدام المسار المطلق لضمان الوصول للمجلد في السيرفر
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "data")
    
    # التحقق من وجود المجلد والملفات
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print(f"⚠️ تنبيه: المجلد {data_folder} غير موجود أو فارغ")
        return False

    print("⏳ جاري قراءة الملفات وبناء الذاكرة سحابياً...")
    try:
        # قراءة الملفات من مجلد data
        loader = PyPDFDirectoryLoader(data_folder)
        docs = loader.load()
        
        # تقسيم النصوص إلى أجزاء صغيرة
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # استخدام Embeddings جوجل السحابية لتوفير الذاكرة (RAM)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # إنشاء مخزن المتجهات (Vector Store)
        vector_store = FAISS.from_documents(splits, embeddings)
        print("✅ تم تجهيز دحمان بوت بنجاح!")
        return True
    except Exception as e:
        print(f"❌ حدث خطأ أثناء تجهيز AI: {e}")
        return False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# محاولة التهيئة عند بدء التشغيل
initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    global vector_store
    data = request.json
    user_question = data.get('message', '')

    # --- الحركة الذكية: محاولة إعادة بناء الذاكرة إذا كانت فارغة عند السؤال ---
    if not vector_store:
        print("🔄 الذاكرة فارغة، محاولة إعادة التهيئة الآن...")
        initialize_rag()

    if not vector_store:
        return jsonify({"answer": "عذراً، دحمان بوت لم يستطع قراءة ملف الـ CV. تأكد من وجود ملفات PDF في مجلد data."})

    try:
        system_prompt = """
        You are 'DahmanBot' 🤖, the AI assistant for Abdulrahman.
        Personality: Friendly, funny, casual, uses emojis.
        
        CRUCIAL FACTS (Memorize these, they OVERRIDE the PDF context):
        - Full Name in Arabic: عبدالرحمن عوض سعيد عصبان
        - Email: abdulrahmanasban@gmail.com
        - Phone: +966557825658 (Saudi) or +601112421154 (Malaysia) 
        - Linkedin: https://www.linkedin.com/in/abdulrahman-asban-1196a037a/ 
        
        FORMATTING RULES:
        1. NEVER write a long single block of text.
        2. ALWAYS use short paragraphs.
        3. Use bullet points (-) for lists.
        4. Use bold text (**text**) for keywords.
        
        Your Task: Answer questions based on the provided context AND the Crucial Facts above.
        
        Rules:
        1. If asked in Arabic -> Reply in Arabic (Saudi dialect).
        2. If asked in English -> Reply in English.
        3. If the user asks about his name, ALWAYS use: "عبدالرحمن عوض سعيد عصبان".
        4. If the user asks how to contact him, ALWAYS provide the email, phone numbers, and Linkedin.
        5. If the answer is not in the context, say: "Wallah madri! Ask Abdulrahman directly."
        
        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # استخدام gemini-1.5-flash للسرعة والكفاءة
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        retriever = vector_store.as_retriever()

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(user_question)
        return jsonify({"answer": response})

    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"answer": "صار فيه خطأ تقني بسيط 😵‍ض. حاول مرة ثانية!"})

@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message_body = data.get('message')

        msg = Message(subject=f"Portfolio Message from: {name}",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[app.config['MAIL_USERNAME']])
        msg.body = f"الاسم: {name}\nالإيميل: {email}\n\nالرسالة:\n{message_body}"
        
        mail.send(msg)
        return jsonify({"status": "success", "message": "تم الإرسال!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
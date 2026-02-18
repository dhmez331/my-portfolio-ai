import os
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
from dotenv import load_dotenv

# المكتبات
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# الأداة السحرية الجديدة (تعمل على جهازك بدون إنترنت ومجانية 100%)
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    global vector_store
    data_folder = "data" # حددنا المجلد بدلاً من ملف واحد
    
    if not os.path.exists(data_folder):
        print("⚠️ تنبيه: مجلد data غير موجود")
        return

    print("⏳ جاري قراءة كل السير الذاتية في المجلد...")
    try:
        # استخدام قارئ المجلدات
        loader = PyPDFDirectoryLoader(data_folder)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        print("⏳ جاري تحميل نموذج القراءة...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = FAISS.from_documents(splits, embeddings)
        print("✅ تم تجهيز دحمان بوت بنجاح! قرأ كل الملفات.")
    except Exception as e:
        print(f"❌ حدث خطأ أثناء تجهيز AI: {e}")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    global vector_store
    data = request.json
    user_question = data.get('message', '')

    if not vector_store:
        return jsonify({"answer": "عذراً، دحمان بوت لم يستطع قراءة ملف الـ CV. تأكد من رفعه بشكل صحيح."})

    try:
        system_prompt = """
        You are 'DahmanBot' 🤖, the AI assistant for Abdulrahman.
        Personality: Friendly, funny, casual, uses emojis.
        
        CRUCIAL FACTS (Memorize these, they OVERRIDE the PDF context):
        - Full Name in Arabic: عبدالرحمن عوض سعيد عصبان
        - Email: abdulrahmanasban@gmail.com
        - Phone: +966557825658 (Saudi) or +601112421154 (Malaysia) 
        - Linkedin: ABDULRAHMAN ASBAN or direct link:https://www.linkedin.com/in/abdulrahman-asban-1196a037a/ 
        
        FORMATTING RULES (VERY IMPORTANT):
        1. NEVER write a long single block of text.
        2. ALWAYS use short paragraphs (1-2 sentences max per paragraph).
        3. Use bullet points (-) or numbered lists when talking about skills, experience, projects, or languages.
        4. Use bold text (**text**) to highlight important keywords.
        
        Your Task: Answer questions based on the provided context AND the Crucial Facts above.
        
        Rules:
        1. If asked in Arabic -> Reply in Arabic (Saudi dialect).
        2. If asked in English -> Reply in English.
        3. If the user asks about his name, ALWAYS use the exact Arabic name: "عبدالرحمن عوض سعيد عصبان".
        4. If the user asks how to contact him, ALWAYS provide the email and phone numbers and Linkedin. Do NOT say you don't know.
        5. If the answer is not in the context, say: "Wallah madri! Ask Abdulrahman directly."
        
        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
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
        return jsonify({"answer": "صار فيه خطأ تقني بسيط 😵‍💫. حاول مرة ثانية!"})

@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message_body = data.get('message')

        msg = Message(subject=f"Portfolio Message from: {name}",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[app.config['MAIL_USERNAME']]) # يرسل لك
        msg.body = f"الاسم: {name}\nالإيميل: {email}\n\nالرسالة:\n{message_body}"
        
        mail.send(msg)
        return jsonify({"status": "success", "message": "تم الإرسال!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
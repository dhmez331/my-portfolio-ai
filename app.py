import os
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
from dotenv import load_dotenv

# استدعاء ChatGroq من langchain_groq
from langchain_groq import ChatGroq

# مكتبات RAG
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.schema import HumanMessage, SystemMessage

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
    """تهيئة ذاكرة الذكاء الاصطناعي وقراءة ملفات PDF"""
    global vector_store
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print(f"⚠️ تنبيه: المجلد {data_folder} غير موجود أو فارغ")
        return False

    print("⏳ جاري قراءة الملفات وبناء الذاكرة سحابياً...")
    try:
        loader = PyPDFDirectoryLoader(data_folder)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.from_documents(splits, embeddings)
        
        print("✅ تم تجهيز دحمان بوت بنجاح!")
        return True
    except Exception as e:
        print(f"❌ حدث خطأ أثناء تجهيز AI: {e}")
        return False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# initialize_rag()  # يمكن تشغيلها عند البداية لو تحب

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    global vector_store
    data = request.json
    user_question = data.get('message', '')

    # إعادة تهيئة الذاكرة لو كانت فارغة
    if not vector_store:
        print("🔄 الذاكرة فارغة، محاولة إعادة التهيئة الآن...")
        initialize_rag()

    if not vector_store:
        return jsonify({"answer": "عذراً، دحمان بوت لم يستطع قراءة ملفات PDF. تأكد من وجودها في مجلد data."})

    try:
        # دمج Crucial Facts + شخصية مرحة
        system_prompt = """
        أنت الآن 'دحمان بوت' 🤖، المساعد الشخصي لعبدالرحمن.
        شخصية: مرحة، دعابة خفيفة، وكأنك صديق قديم. استخدم إيموجي.
        
        معلومات مهمة (Crucial Facts):
        - الاسم الكامل: عبدالرحمن عوض سعيد عصبان
        - البريد: abdulrahmanasban@gmail.com
        - الهاتف: +966557825658 (سعودي) أو +601112421154 (ماليزيا)
        - Linkedin: https://www.linkedin.com/in/abdulrahman-asban-1196a037a/
        
        القواعد:
        1. إذا سألوا عن عبدالرحمن، تحدث عنه كأنه شخص ثاني بطريقة مرحة.
        2. استخدم فقرات قصيرة وقوائم نقطية.
        3. إذا السؤال مش موجود في المعلومات، قل: "Wallah madri! Ask Abdulrahman directly."
        4. الرد بالعربية السعودية إذا كان السؤال بالعربي، وبالإنجليزي إذا كان السؤال بالإنجليزي.
        """

        # تهيئة LLM Groq
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )

        def ask_dahman_bot(user_input):
            response = llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            return response.content

        # تجربة قصيرة
        print(ask_dahman_bot("من أنت؟"))
        print(ask_dahman_bot("من هو عبدالرحمن؟"))

        # استخدام الـ RAG
        retriever = vector_store.as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
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
                      recipients=[app.config['MAIL_USERNAME']])
        msg.body = f"الاسم: {name}\nالإيميل: {email}\n\nالرسالة:\n{message_body}"
        
        mail.send(msg)
        return jsonify({"status": "success", "message": "تم الإرسال!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

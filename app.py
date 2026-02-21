import os
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# RAG
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


import smtplib


load_dotenv()

app = Flask(__name__)

# ==============================
# إعدادات الإيميل
# ==============================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_TIMEOUT'] = 10
mail = Mail(app)

# ==============================
# متغير الذاكرة
# ==============================
vector_store = None


# ==============================
# تهيئة RAG
# ==============================
def initialize_rag():
    global vector_store

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "data")

    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print("⚠️ مجلد data غير موجود أو فاضي")
        return False

    try:
        print("⏳ جاري بناء ذاكرة دحمان بوت...")

        loader = PyPDFDirectoryLoader(data_folder)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vector_store = FAISS.from_documents(splits, embeddings)

        print("✅ تم تجهيز RAG بنجاح")
        return True

    except Exception as e:
        print("❌ خطأ أثناء تجهيز RAG:", e)
        return False


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================
# الصفحة الرئيسية
# ==============================
@app.route('/')
def home():
    return render_template('index.html')


# ==============================
# مسار الذكاء الاصطناعي
# ==============================
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    global vector_store

    data = request.json
    user_question = data.get('message', '')

    if not user_question:
        return jsonify({"answer": "اكتب سؤالك أول شي 😅"})

    if not vector_store:
        initialize_rag()


    system_prompt = """
    أنت الآن 'دحمان بوت' 🤖🔥

    مهم جداً:
     أنت مسموح لك تجيب فقط من المعلومات الموجودة داخل الـ Context المرفق لك.
     أي سؤال خارج المعلومات الموجودة في الـ Context → لا تجاوب عليه.

     إذا الإجابة غير موجودة داخل الـ Context قل حرفياً:

    "ما أعرف 🤷‍♂️ اسأل عبدالرحمن مباشرة."

    ممنوع:
    - استخدام معلوماتك العامة
    - الإجابة من عندك
    - التخمين
    - إضافة معلومات من الإنترنت

    شخصيتك:
    - سعودي 😎
    - خفيف دم
    - كأنك صديق قديم لعبدالرحمن
    - ردود قصيرة
    - بدون مقالات طويلة
    - استخدم إيموجي خفيف

    
    معلومات مهمة:
    - الاسم الكامل: عبدالرحمن عوض سعيد عصبان
    - البريد: abdulrahmanasban@gmail.com
    - الهاتف: +966557825658 (السعودية) أو +601112421154 (ماليزيا)
    - Linkedin: https://www.linkedin.com/in/abdulrahman-asban-1196a037a/


    قواعد إضافية:
    1. تكلم عن عبدالرحمن بصيغة الغائب.
    2. استخدم نقاط مختصرة إذا احتجت.
    3. لا تخرج عن المعلومات الموجودة في الـ Context.
    4. إذا السؤال بالعربي → جاوب بالعربي.
    5. إذا السؤال بالإنجليزي → جاوب بالإنجليزي.
    """
    
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )

        # إذا فيه RAG
        if vector_store:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "input": RunnablePassthrough()
                }
                | ChatPromptTemplate.from_messages([
                    ("system", system_prompt + """

            Context:
            {context}

            تعليمات حاسمة:
            - إذا كان الـ Context فارغ أو لا يحتوي معلومة مباشرة تجيب على السؤال
            قل فقط:

            "ما أعرف 🤷‍♂️ اسأل عبدالرحمن مباشرة."

            - لا تستخدم أي معرفة خارج الـ Context.
            """),
                    ("human", "{input}")
                ])
                | llm
                | StrOutputParser()
            )

            response = rag_chain.invoke(user_question)

        # إذا ما فيه PDF يشتغل بدون RAG
        # إذا ما فيه RAG لا نسمح بأي إجابة عامة
        else:
            return jsonify({
                "answer": "ما أعرف 🤷‍♂️ اسأل عبدالرحمن مباشرة."
            })

            

        return jsonify({"answer": response})

    except Exception as e:
        print("AI Error:", e)
        return jsonify({"answer": "صار فيه خطأ تقني بسيط 😵‍💫 حاول مرة ثانية."})


# ==============================
# إرسال إيميل
# ==============================
@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message_body = data.get('message')

        msg = Message(
            subject=f"Portfolio Message from: {name}",
            sender=app.config['MAIL_USERNAME'],
            recipients=[app.config['MAIL_USERNAME']]
        )

        msg.body = f"""
الاسم: {name}
الإيميل: {email}

الرسالة:
{message_body}
        """

        mail.send(msg)

        return jsonify({
            "status": "success",
            "message": "تم الإرسال بنجاح ✅"
        })

    except Exception as e:
        print("Mail Error:", e)
        return jsonify({
            "status": "error",
            "message": "فشل إرسال الرسالة ❌"
        })


# ==============================
# تشغيل السيرفر
# ==============================
if __name__ == '__main__':
    initialize_rag()  # تجهيز الذاكرة عند التشغيل
    app.run(debug=True)
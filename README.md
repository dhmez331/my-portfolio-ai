# 🌐 Abdulrahman Asban — Personal Portfolio & AI Chatbot

A personal portfolio website with an integrated RAG-powered chatbot trained on my CV, built with Flask and deployed on Render.

---

## 🚀 Live Demo

🔗 [abdulrahman-portfolio-ai.onrender.com](https://abdulrahman-portfolio-ai.onrender.com)

---

## ✨ Features

- 🤖 **DahmanBot** — AI chatbot powered by RAG (Retrieval-Augmented Generation), answers questions only from CV data
- 🌍 **Bilingual** — Full Arabic & English support with RTL/LTR switching
- 🌙 **Dark / Light Mode** — Theme toggle with persistent preference
- 📩 **Contact Form** — Sends email directly via EmailJS (no backend needed)
- 🎨 **Animated UI** — Particles background, typing effect, scroll animations
- 📱 **Responsive** — Works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| AI / LLM | Groq (LLaMA 3.3 70B) |
| Embeddings | Google Gemini Embedding |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Email | EmailJS |
| Hosting | Render |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
portfolio/
│
├── app.py                  # Flask server & all routes
├── requirements.txt        # Python dependencies
├── .env                    # Secret keys (not tracked)
│
├── data/
│   └── resume_ar.pdf              # CV used for RAG
│   └── resume_en.pdf
|
├── static/
│   ├── style.css
│   ├── script.js
│   └── resume_en.pdf       # Downloadable CV
│
└── templates/
    └── index.html
```

---

## ⚙️ How It Works

### On server startup:
1. Reads the CV PDF from `/data`
2. Splits it into chunks
3. Builds a FAISS vector store using Gemini embeddings
4. RAG chain is ready

### On `/ask_ai` request:
1. Receives the user's question + language preference
2. Retrieves relevant chunks from the vector store
3. Sends context + question to LLaMA via Groq
4. Returns an answer strictly based on CV data

### Contact Form:
- Handled entirely on the frontend via EmailJS
- No SMTP or backend required

---

## 🔧 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/dhmez331/my-portfolio-ai.git
cd portfolio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key


# 4. Add your CV
# Place your CV PDF inside the /data folder

# 5. Run the server
python app.py
```

---

## 🌍 Deployment (Render)

1. Push code to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn --timeout 120 --bind 0.0.0.0:$PORT app:app`
5. Add environment variables from `.env`
6. Deploy 🎉

> **Tip:** Use [UptimeRobot](https://uptimerobot.com) to ping your service every 5 minutes and prevent it from sleeping on the free tier.

---

## 👨‍💻 Author

**Abdulrahman Asban**
- 📧 abdulrahmanasban@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/abdulrahman-asban-1196a037a/)
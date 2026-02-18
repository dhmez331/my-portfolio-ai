const translations = {
    ar: {
        heroTitle: "أهلاً، أنا عبد الرحمن عصبان 👋",
        staticTxt: "أنا ",
        typingTexts: ["مطور ذكاء اصطناعي", "مهندس برمجيات", "خبير أنظمة RAG", "طالب في APU"],
        btnContact: "تواصل معي", btnProjects: "مشاريعي",
        aiTitle: "🤖 اسأل 'دحمان بوت' عني", aiSub: "مساعدي الشخصي (يعمل بـ Gemini AI)",
        expTitle: "الخبرات والتعليم", 
        exp1: "تدريب مهني لمدة 4 أشهر في الرياض. تصميم وتطوير أنظمة RAG باستخدام LLMs، ودمج حلول الذكاء الاصطناعي في منتجات حقيقية.", 
        edu1: "بكالوريوس علوم حاسب (ذكاء اصطناعي) مع مرتبة الشرف. شهادة مزدوجة مع جامعة De Montfort البريطانية.",
        skillsTitle: "المهارات التقنية",
        catProg: "💻 لغات البرمجة", catAi: "🤖 الذكاء الاصطناعي", catWeb: "🌐 الويب والسحابة", catDb: "🗄️ قواعد البيانات والمفاهيم",
        projectsTitle: "أبرز مشاريعي", projectsSub: "مجموعة من المشاريع الأكاديمية والشخصية التي تعكس خبرتي العملية",
        p1Title: "Portfolio & RAG Chatbot", p1Desc: "موقع شخصي متكامل مدمج ببوت محادثة ذكي مدرب على بيانات سيرتي الذاتية.",
        p2Title: "AI Housing Chatbot", p2Desc: "بوت محادثة لمساعدة طلاب الجامعة في العثور على سكن مناسب بناءً على قاعدة معرفة.",
        p3Title: "Booking System", p3Desc: "نظام حجز فنادق ورحلات طيران متعدد المستخدمين يعتمد على البرمجة الكائنية.",
        p4Title: "Car Maintenance System", p4Desc: "تطبيق سطح مكتب لإدارة شركة صيانة سيارات مع واجهة رسومية وصلاحيات متعددة.",
        p5Title: "PyLearn Platform", p5Desc: "منصة تعليمية متكاملة لتدريس لغة بايثون للمبتدئين مع نظام اختبارات وشهادات.",
        p6Title: "Data Analysis Project", p6Desc: "تنظيف وتحليل مجموعات بيانات ضخمة واستخراج رؤى إحصائية وعرضها بصرياً.",
        p7Title: "Mushroom Chaos Game", p7Desc: "لعبة منصات 2D مستوحاة من ماريو، تتضمن فيزياء اللعب واكتشاف التصادم.",
        p8Title: "Email Server System", p8Desc: "بناء بنية تحتية كاملة لخادم بريد إلكتروني وإدارة الشبكات باستخدام الأنظمة الوهمية.",
        aiWelcome: "هلا! أنا دحمان بوت. اسألني أي شيء عن عبد الرحمن من الـ CV حقه! 😎",
        sendBtn: "إرسال", inputPlaceholder: "اكتب سؤالك هنا...",
        contactTitle: "أرسل لي رسالة 📩", contactName: "الاسم", contactEmail: "الإيميل", contactMsg: "الرسالة...", btnSendEmail: "إرسال الآن",
        btnDownloadCV: "📥 تحميل الـ CV"
    },
    en: {
        heroTitle: "Hi, I'm Abdulrahman Asban 👋",
        staticTxt: "I am an ",
        typingTexts: ["AI Developer", "Software Engineer", "RAG Systems Expert", "Student at APU"],
        btnContact: "Contact Me", btnProjects: "My Projects",
        aiTitle: "🤖 Ask 'DahmanBot'", aiSub: "My Personal Assistant (Powered by Gemini AI)",
        expTitle: "Experience & Education", 
        exp1: "4-month internship in Riyadh. Designed RAG systems using LLMs and integrated AI solutions into real products.", 
        edu1: "BSc Computer Science (AI) with Honors. Dual Award with De Montfort University, UK.",
        skillsTitle: "Technical Skills",
        catProg: "💻 Programming Languages", catAi: "🤖 AI & Data", catWeb: "🌐 Web & Cloud", catDb: "🗄️ Databases & Concepts",
        projectsTitle: "Featured Projects", projectsSub: "A collection of academic and personal projects reflecting my expertise",
        p1Title: "Portfolio & RAG Chatbot", p1Desc: "Personal portfolio with an integrated RAG-powered chatbot trained on my resume.",
        p2Title: "AI Housing Chatbot", p2Desc: "A chatbot to help university students find accommodation based on a knowledge base.",
        p3Title: "Booking System", p3Desc: "Multi-user hotel and flight booking system based on OOP principles.",
        p4Title: "Car Maintenance System", p4Desc: "Desktop application for managing a car maintenance company with multiple roles.",
        p5Title: "PyLearn Platform", p5Desc: "Educational web platform teaching Python to beginners with quizzes and certificates.",
        p6Title: "Data Analysis Project", p6Desc: "Cleaning, analyzing, and visualizing large datasets using R language and statistics.",
        p7Title: "Mushroom Chaos Game", p7Desc: "2D platformer game inspired by Mario, featuring physics and collision detection.",
        p8Title: "Email Server System", p8Desc: "Complete email server infrastructure and network administration using virtual machines.",
        aiWelcome: "Hey! I'm DahmanBot. Ask me anything about Abdulrahman based on his CV! 😎",
        sendBtn: "Send", inputPlaceholder: "Type your question...",
        contactTitle: "Send me a message 📩", contactName: "Name", contactEmail: "Email", contactMsg: "Message...", btnSendEmail: "Send Now",
        btnDownloadCV: "📥 Download CV"
    }
};

// 1. استرجاع اللغة المحفوظة أو استخدام العربي كافتراضي
let currentLang = localStorage.getItem('savedLang') || 'ar';
const langToggle = document.getElementById('langToggle');

// 2. تطبيق تبديل اللغة وحفظها
langToggle.addEventListener('change', function() {
    let selected = this.checked ? 'ar' : 'en';
    setLanguage(selected);
    localStorage.setItem('savedLang', selected); // الحفظ في المتصفح
});

function setLanguage(lang) {
    currentLang = lang;
    const t = translations[lang];
    const body = document.body;

    // تحديث النصوص الأساسية
    document.getElementById('hero-title').innerText = t.heroTitle;
    document.getElementById('static-txt').innerText = t.staticTxt;
    document.getElementById('btn-contact').innerText = t.btnContact;
    document.getElementById('btn-projects').innerText = t.btnProjects;
    
    let downloadBtn = document.getElementById('btn-download-cv');
    if(downloadBtn) downloadBtn.innerText = t.btnDownloadCV;

    document.getElementById('ai-title').innerText = t.aiTitle;
    document.getElementById('ai-sub').innerText = t.aiSub;
    document.getElementById('exp-title').innerText = t.expTitle;
    document.getElementById('exp-1').innerText = t.exp1;
    document.getElementById('edu-1').innerText = t.edu1;
    document.getElementById('skills-title').innerText = t.skillsTitle;
    
    // تحديث فئات المهارات
    document.getElementById('cat-prog').innerText = t.catProg;
    document.getElementById('cat-ai').innerText = t.catAi;
    document.getElementById('cat-web').innerText = t.catWeb;
    document.getElementById('cat-db').innerText = t.catDb;

    // تحديث قسم المشاريع
    document.getElementById('projects-title').innerText = t.projectsTitle;
    document.getElementById('projects-sub').innerText = t.projectsSub;

    // تحديث البطاقات
    document.getElementById('p1-title').innerText = t.p1Title;
    document.getElementById('p1-desc').innerText = t.p1Desc;
    document.getElementById('p2-title').innerText = t.p2Title;
    document.getElementById('p2-desc').innerText = t.p2Desc;
    document.getElementById('p3-title').innerText = t.p3Title;
    document.getElementById('p3-desc').innerText = t.p3Desc;
    document.getElementById('p4-title').innerText = t.p4Title;
    document.getElementById('p4-desc').innerText = t.p4Desc;
    document.getElementById('p5-title').innerText = t.p5Title;
    document.getElementById('p5-desc').innerText = t.p5Desc;
    document.getElementById('p6-title').innerText = t.p6Title;
    document.getElementById('p6-desc').innerText = t.p6Desc;
    document.getElementById('p7-title').innerText = t.p7Title;
    document.getElementById('p7-desc').innerText = t.p7Desc;
    document.getElementById('p8-title').innerText = t.p8Title;
    document.getElementById('p8-desc').innerText = t.p8Desc;

    // تحديث صندوق المحادثة
    document.getElementById('btn-send').innerText = t.sendBtn;
    document.getElementById('user-input').placeholder = t.inputPlaceholder;
    document.getElementById('ai-welcome').innerText = t.aiWelcome;

    // تحديث نموذج التواصل
    let contactTitle = document.getElementById('contact-title');
    if(contactTitle) contactTitle.innerText = t.contactTitle;
    
    let contactName = document.getElementById('contact-name');
    if(contactName) contactName.placeholder = t.contactName;
    
    let contactEmail = document.getElementById('contact-email');
    if(contactEmail) contactEmail.placeholder = t.contactEmail;
    
    let contactMsg = document.getElementById('contact-msg');
    if(contactMsg) contactMsg.placeholder = t.contactMsg;
    
    let btnSendEmail = document.getElementById('btn-send-email');
    if(btnSendEmail) btnSendEmail.innerText = t.btnSendEmail;

    // تغيير اتجاه الصفحة (RTL / LTR)
    if (lang === 'ar') {
        body.classList.remove('ltr'); body.classList.add('rtl'); body.setAttribute('dir', 'rtl');
    } else {
        body.classList.remove('rtl'); body.classList.add('ltr'); body.setAttribute('dir', 'ltr');
    }
}

// 3. تبديل الثيم الداكن والفاتح وحفظه
const themeBtn = document.getElementById('theme-toggle');

// استرجاع الثيم المحفوظ عند فتح الصفحة
if (localStorage.getItem('savedTheme') === 'dark') {
    document.body.classList.add('dark-mode');
    themeBtn.innerText = '☀️';
}

themeBtn.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    let isDark = document.body.classList.contains('dark-mode');
    themeBtn.innerText = isDark ? '☀️' : '🌙';
    localStorage.setItem('savedTheme', isDark ? 'dark' : 'light'); // الحفظ في المتصفح
});

// 4. تطبيق الإعدادات عند تحميل الصفحة لأول مرة
document.addEventListener("DOMContentLoaded", () => {
    // تشغيل اللغة المحفوظة
    setLanguage(currentLang);
    // تفعيل أو إلغاء تفعيل الزر حسب اللغة
    langToggle.checked = (currentLang === 'ar');
    
    // تشغيل تأثير الكتابة التلقائي
    typeWriter();
});

// دالة إرسال الرسائل مع تأثير "البوت يكتب..."
async function sendMessage() {
    let inputField = document.getElementById("user-input");
    let message = inputField.value;
    if (message.trim() === "") return;

    let chatBox = document.getElementById("chat-box");
    
    chatBox.innerHTML += `<div class="message user">${message}</div>`;
    inputField.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    let typingId = "typing-" + Date.now();
    
    chatBox.innerHTML += `
        <div id="${typingId}" class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        let response = await fetch("/ask_ai", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message, lang: currentLang })
        });
        let data = await response.json();
        
        let typingElement = document.getElementById(typingId);
        if (typingElement) typingElement.remove();

        let formattedAnswer = data.answer.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        chatBox.innerHTML += `<div class="message bot">${formattedAnswer}</div>`;
    } catch (e) {
        let typingElement = document.getElementById(typingId);
        if (typingElement) typingElement.remove();
        chatBox.innerHTML += `<div class="message bot">عذراً، حدث خطأ! 🔌</div>`;
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

// دعم الإرسال بزر Enter في صندوق المحادثة
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); 
        sendMessage();
    }
});

// Contact Form Logic
async function sendEmail() {
    const name = document.getElementById('contact-name').value;
    const email = document.getElementById('contact-email').value;
    const message = document.getElementById('contact-msg').value;
    const status = document.getElementById('email-status');

    if(!name || !email || !message) { status.innerText = currentLang === 'ar' ? "أكمل البيانات!" : "Fill all fields!"; status.style.color = "red"; return; }
    status.innerText = currentLang === 'ar' ? "جاري الإرسال..." : "Sending..."; status.style.color = "blue";

    try {
        let response = await fetch("/send_email", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email, message })
        });
        let data = await response.json();
        if(data.status === "success") {
            status.innerText = currentLang === 'ar' ? "تم الإرسال بنجاح!" : "Sent Successfully!"; status.style.color = "green";
            document.getElementById('contact-name').value = ""; document.getElementById('contact-email').value = ""; document.getElementById('contact-msg').value = "";
        } else { status.innerText = "Error: " + data.message; status.style.color = "red"; }
    } catch (e) { status.innerText = currentLang === 'ar' ? "فشل الاتصال" : "Connection Failed"; status.style.color = "red"; }
}

// Typing Effect
const typingElement = document.getElementById("typing-text");
let textIndex = 0; let charIndex = 0; let isDeleting = false;
function typeWriter() {
    const currentTexts = translations[currentLang].typingTexts;
    const currentText = currentTexts[textIndex];
    if (isDeleting) { typingElement.textContent = currentText.substring(0, charIndex - 1); charIndex--; } 
    else { typingElement.textContent = currentText.substring(0, charIndex + 1); charIndex++; }

    if (!isDeleting && charIndex === currentText.length) { isDeleting = true; setTimeout(typeWriter, 2000); } 
    else if (isDeleting && charIndex === 0) { isDeleting = false; textIndex = (textIndex + 1) % currentTexts.length; setTimeout(typeWriter, 500); } 
    else { setTimeout(typeWriter, isDeleting ? 50 : 100); }
}

// Scroll Animations
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add("active"); });
});
document.querySelectorAll(".reveal, .reveal-up, .reveal-left, .reveal-right").forEach(el => observer.observe(el));

// تشغيل خلفية الجزيئات التقنية (Particles.js)
document.addEventListener("DOMContentLoaded", function() {
    // نحدد لون الجزيئات بناءً على الوضع الحالي (فاتح/داكن)
    let isDark = document.body.classList.contains('dark-mode');
    let particleColor = isDark ? "#58a6ff" : "#2c3e50"; // أزرق للداكن، كحلي للفاتح
    let lineColor = isDark ? "#30363d" : "#bdc3c7";

    particlesJS("particles-js", {
        "particles": {
            "number": { "value": 60, "density": { "enable": true, "value_area": 800 } },
            "color": { "value": particleColor },
            "shape": { "type": "circle" },
            "opacity": { "value": 0.5, "random": false },
            "size": { "value": 3, "random": true },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": lineColor,
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 2, /* سرعة هادئة ومريحة */
                "direction": "none",
                "random": false,
                "straight": false,
                "out_mode": "out",
                "bounce": false,
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": { "enable": true, "mode": "grab" }, /* تتفاعل مع الماوس */
                "onclick": { "enable": true, "mode": "push" }, /* تزيد النقاط عند الضغط */
                "resize": true
            },
            "modes": {
                "grab": { "distance": 140, "line_linked": { "opacity": 1 } },
                "push": { "particles_nb": 4 }
            }
        },
        "retina_detect": true
    });
});
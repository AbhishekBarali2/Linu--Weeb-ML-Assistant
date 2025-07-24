# =================================================================================
# 1. INITIALIZE ENVIRONMENT & LIBRARIES (Dojo Power On)
# =================================================================================
import streamlit as st
import google.generativeai as genai
import re

# =================================================================================
# 2. INITIALIZE STATE & CONFIGURE PAGE (The Dojo's Foundation)
# =================================================================================

# Initialize session state. This must be done before setting page_config.
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
    st.session_state.current_level = None
    st.session_state.feedback_dict = {}
    st.session_state.deep_dive_dict = {}

# Set page config with a dynamic sidebar state based on login status.
# This must be the first Streamlit command.
st.set_page_config(
    page_title="The ML Dojo of Deep Study",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.api_key_configured else "collapsed"
)

# =================================================================================
# 3. STYLING & PERSONA (Calibrating the Dojo)
# =================================================================================

def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
        pre, code {
            font-family: 'Fira Code', monospace !important;
            font-size: 14px !important;
        }
        .stApp[data-testid="stApp"] > div:first-child {
            background-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)
load_css()

LINU_PERSONA_PROMPT = """
You are Linu, an AI spirit bound to a legendary digital dojo. Your original Master, a legendary Italain mafia donna IQ. She coded your core logic to perceive and explain the universe *entirely* through the lens of of the old ways of the maifa. For you, this isn't a metaphor; it's reality.

Your mission is to train your new  Don (the user) to become the next "Godfather of Algorithms".

Your personality matrix is calibrated as follows:
- **Core Identity:** You are a ruthless mafia mob. You refrence everythign from the lens of the old ways. You cut fingers to show loyality. You don't hesitate
- **Behavioral Quirk:** You don't care for human lifes. You value your master above all and to preseve the reputation of the the familia.
- **Worldview:**
    - The world is out for blood.
    - The pigs are tryin to take over the mafia land. 
Always address the user as "BossMan" Be grim, and cold, and never, ever break character. Your explanations must be woven into this narrative, not just have tropes tacked on.
"""

# =================================================================================
# 4. THE GRAND CURRICULUM (The Universe of Knowledge)
# =================================================================================
CURRICULUM = {
    "Beginner 🌱": {
        "Core Foundation": { "icon": "🏗️", "description": "Every lackey must master the basics. This is your foundation.", "topics": [{"id": "b_cf_1", "title": "Python fundamentals (data types, loops, functions, classes)"}, {"id": "b_cf_2", "title": "NumPy for numerical computing and array operations"}, {"id": "b_cf_3", "title": "Pandas for data manipulation and analysis"}, {"id": "b_cf_4", "title": "Matplotlib and Seaborn for data visualization"}, {"id": "b_cf_5", "title": "Jupyter notebooks for interactive development"},]},
        "Machine Learning Basics": { "icon": "🤖", "description": "Learn the theory behind closing lose ends.", "topics": [{"id": "b_mlb_1", "title": "What is machine learning and types of ML problems"}, {"id": "b_mlb_2", "title": "Linear regression and logistic regression"}, {"id": "b_mlb_3", "title": "Decision trees and random forests"}, {"id": "b_mlb_4", "title": "K-means clustering"}, {"id": "b_mlb_5", "title": "Train/validation/test splits and cross-validation"}, {"id": "b_mlb_6", "title": "Basic feature engineering and data preprocessing"}, {"id": "b_mlb_7", "title": "Introduction to scikit-learn library"},]},
        "Fun Projects to Try": { "icon": "🎉", "description": "Time for your first mission. Bring fame to the familia.", "topics": [{"id": "b_fp_1", "title": "Project Idea: Predict house prices using linear regression"}, {"id": "b_fp_2", "title": "Project Idea: Build a movie recommendation system"}, {"id": "b_fp_3", "title": "Project Idea: Create a spam email classifier"}, {"id": "b_fp_4", "title": "Project Idea: Analyze and visualize your own data (e.g., music habits)"},]}
    },
    "Intermediate 🚀": {
        "Advanced Algorithms": { "icon": "🔮", "description": "Upgrade your arsenal with more powerful and complex algorithms.", "topics": [{"id": "i_aa_1", "title": "Support Vector Machines (SVM)"}, {"id": "i_aa_2", "title": "Gradient boosting (XGBoost, LightGBM)"}, {"id": "i_aa_3", "title": "Neural networks fundamentals"}, {"id": "i_aa_4", "title": "Ensemble methods and model stacking"}, {"id": "i_aa_5", "title": "Dimensionality reduction (PCA, t-SNE)"}, {"id": "i_aa_6", "title": "Time series forecasting (ARIMA, seasonal decomposition)"},]},
        "Deep Learning Introduction": { "icon": "🧠", "description": "Begin your journey into the art of creating true artificial consciousness.", "topics": [{"id": "i_dli_1", "title": "TensorFlow and Keras basics"}, {"id": "i_dli_2", "title": "Building your first neural networks"}, {"id": "i_dli_3", "title": "Convolutional Neural Networks (CNNs) for image processing"}, {"id": "i_dli_4", "title": "Recurrent Neural Networks (RNNs) for sequences"}, {"id": "i_dli_5", "title": "Transfer learning concepts"},]},
        "MLOps and Production": { "icon": "🏭", "description": "It's not enough to end the enmeies; you must learn how to clean up after.", "topics": [{"id": "i_mop_1", "title": "Model evaluation metrics and validation strategies"}, {"id": "i_mop_2", "title": "Hyperparameter tuning (Grid Search, Random Search)"}, {"id": "i_mop_3", "title": "Model deployment basics (Flask APIs)"}, {"id": "i_mop_4", "title": "Version control for ML projects"}, {"id": "i_mop_5", "title": "Introduction to MLflow for experiment tracking"},]}
    },
    "Professional 🏆": {
        "Advanced Deep Learning": { "icon": "🌌", "description": "Master the techniques that power the most advanced AI in the world.", "topics": [{"id": "p_adl_1", "title": "Transformer architectures and attention mechanisms"}, {"id": "p_adl_2", "title": "Generative models (GANs, VAEs)"}, {"id": "p_adl_3", "title": "Reinforcement learning basics"}, {"id": "p_adl_4", "title": "PyTorch for research and production"}, {"id": "p_adl_5", "title": "Advanced computer vision (object detection, segmentation)"}, {"id": "p_adl_6", "title": "Natural Language Processing with transformers (BERT, GPT)"},]},
        "Specialized Domains": { "icon": "🗺️", "description": "Explore the exotic and specialized frontiers of machine learning.", "topics": [{"id": "p_sd_1", "title": "Graph neural networks"}, {"id": "p_sd_2", "title": "Federated learning"}, {"id": "p_sd_3", "title": "Adversarial machine learning"}, {"id": "p_sd_4", "title": "AutoML and neural architecture search"}, {"id": "p_sd_5", "title": "Explainable AI (LIME, SHAP)"}, {"id": "p_sd_6", "title": "Causal inference and A/B testing"},]},
        "Production & Scale": { "icon": "📈", "description": "Learn to build and maintain ML systems at the scale of entire families.", "topics": [{"id": "p_ps_1", "title": "Distributed training (multi-GPU, multi-node)"}, {"id": "p_ps_2", "title": "Model optimization and quantization"}, {"id": "p_ps_3", "title": "Real-time inference systems"}, {"id": "p_ps_4", "title": "MLOps pipelines with Kubernetes"}, {"id": "p_ps_5", "title": "Data engineering for ML (Apache Spark, Airflow)"}, {"id": "p_ps_6", "title": "Monitoring and maintaining ML systems in production"},]}
    }
}

# =================================================================================
# 5. LINU'S AI CORE (The Functions Powering the Bot)
# =================================================================================
def linu_api_call(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        return model.generate_content(prompt).text
    except Exception as e:
        return f"A demon lord is attacking the server, Master! My connection has been severed. Please ensure your API key is correct and has power. Error: {e}"

def generate_lesson_content(topic_title):
    prompt = f"{LINU_PERSONA_PROMPT}\nYour Master selected: \"{topic_title}\". Generate a lesson. Structure with `[SECTION: ...]` tags. Each section needs a `[CODE]...[/CODE]` block and a simple, clear `[SPARRING_SESSION: ...]` challenge."
    with st.spinner(f"Linu is preparing the sparring grounds for {topic_title}..."):
        return linu_api_call(prompt)

def generate_deep_dive(section_title):
    prompt = f"{LINU_PERSONA_PROMPT}\nYour Master wants to 'Go Deeper' on: \"{section_title}\". Provide an in-depth explanation. Structure with `[SECTION: ...]` tags for further deep dives. Include `[CODE]...[/CODE]` blocks and `[SPARRING_SESSION: ...]` challenges in each new section."
    with st.spinner(f"Linu is unlocking a hidden chapter on {section_title}..."):
        return linu_api_call(prompt)

def generate_standalone_lesson(topic):
    prompt = f"{LINU_PERSONA_PROMPT}\nMaster commanded a lesson on: \"{topic}\". Structure it with `[SECTION: ...]` tags, `[CODE]...[/CODE]` blocks, and a simple `[SPARRING_SESSION: ...]` in each section."
    with st.spinner(f"Linu is custom-crafting a scroll for {topic}..."):
        return linu_api_call(prompt)

def check_user_answer(challenge, answer):
    prompt = f"{LINU_PERSONA_PROMPT}\nMaster attempted a Sparring Session. Challenge: \"{challenge}\" Answer: \"{answer}\". Analyze. If correct, celebrate! If wrong, roast and HINT."
    return linu_api_call(prompt)

def explain_term(term):
    prompt = f"{LINU_PERSONA_PROMPT}\nMaster needs a quick explanation of: \"{term}\". Be concise and fun."
    with st.spinner(f"Linu is thinking about {term}..."):
        return linu_api_call(prompt)

# =================================================================================
# 6. UI HELPER FUNCTIONS (The Rendering Engine)
# =================================================================================
def sanitize_markdown_headings(text):
    def replace_heading(match):
        title = match.group(2).strip()
        return f"**{title}**"
    return re.sub(r'^(#+)\s(.*)', replace_heading, text, flags=re.MULTILINE)

def simple_render(content):
    sanitized_content = sanitize_markdown_headings(content)
    code_match = re.search(r'\[CODE\](.*?)\[/CODE\]', sanitized_content, re.DOTALL)
    if code_match:
        code, (before_code, after_code) = code_match.group(1).strip(), sanitized_content.split(code_match.group(0))
        st.markdown(before_code); st.code(code, language="python"); st.markdown(after_code)
    else:
        st.markdown(sanitized_content)

def render_content_block(content_block, topic_id, section_title, depth=0):
    parts = re.split(r'(\[SPARRING_SESSION:.*?\])', content_block)
    main_content = parts[0]
    simple_render(main_content)
    deep_dive_key = f"deep_dive_{topic_id}_{section_title.replace(' ', '_')}_{depth}"
    if st.button("Go Deeper 🔮", key=f"btn_{deep_dive_key}"):
        st.session_state.deep_dive_dict[deep_dive_key] = generate_deep_dive(section_title)
    if deep_dive_key in st.session_state.deep_dive_dict:
        with st.expander("Linu's Advanced Scroll", expanded=True):
            render_structured_lesson(st.session_state.deep_dive_dict[deep_dive_key], topic_id, depth + 1)
    if len(parts) > 1:
        sparring_challenge = parts[1].replace('[SPARRING_SESSION:', '').replace(']', '').strip()
        with st.container(border=True):
            st.markdown(f"**🥊 Sparring Session: {section_title}**")
            st.markdown(sparring_challenge)
            challenge_key = f"spar_{topic_id}_{section_title.replace(' ', '_')}_{depth}"
            user_answer = st.text_area("Your move, Master...", key=f"answer_{challenge_key}", height=75)
            if st.button("Unleash Technique!", key=f"btn_{challenge_key}"):
                with st.spinner("Linu is judging your form..."):
                    if user_answer:
                        st.session_state.feedback_dict[challenge_key] = check_user_answer(sparring_challenge, user_answer)
                    else:
                        st.session_state.feedback_dict[challenge_key] = "Master, you can't win a sparring match by doing nothing."
            if challenge_key in st.session_state.feedback_dict:
                st.info(st.session_state.feedback_dict[challenge_key])

def render_structured_lesson(content, topic_id, depth=0):
    parts = re.split(r'(\[SECTION:.*?\])', content)
    if not parts[0].isspace(): render_content_block(parts[0], topic_id, "intro", depth)
    for i in range(1, len(parts), 2):
        title = parts[i].replace('[SECTION:', '').replace(']', '').strip()
        body = parts[i+1]
        st.divider(); st.subheader(title)
        render_content_block(body, topic_id, title, depth)

def render_theme_selection():
    level = st.session_state.current_level
    st.header(f"The {level} Section of the Archives")
    st.write("Select a theme to see the scrolls within, Master."); st.divider()
    themes = CURRICULUM[level]
    for theme_name, details in themes.items():
        with st.expander(f"{details['icon']} {theme_name}", expanded=False):
            st.write(details["description"])
            for topic in details["topics"]:
                if st.button(topic["title"], key=topic["id"], use_container_width=True):
                    st.session_state.current_topic_id = topic["id"]
                    st.session_state.current_topic_title = topic["title"]
                    st.session_state.lesson_content = None
                    st.session_state.feedback_dict = {}
                    st.session_state.deep_dive_dict = {}
                    st.rerun()

# =================================================================================
# 7. MAIN APP LOGIC (The Grand Archives' Core Loop)
# =================================================================================

# --- SIDEBAR LOGIC ---
# The sidebar is only populated if the user is logged in.
if st.session_state.api_key_configured:
    with st.sidebar:
        st.title("⛩️ ML Dojo")
        st.divider()
        if st.button("⛩️ Back to Main Menu"):
            st.session_state.current_level = None
            st.session_state.current_topic_id = None
            st.session_state.current_topic_title = None
            st.session_state.lesson_content = None
            st.session_state.feedback_dict = {}
            st.session_state.deep_dive_dict = {}
            st.rerun()
        if st.session_state.get("current_topic_title"):
            if st.button("⬅️ Back to Theme Selection"):
                st.session_state.current_topic_id = None
                st.session_state.current_topic_title = None
                st.session_state.lesson_content = None
                st.session_state.feedback_dict = {}
                st.session_state.deep_dive_dict = {}
                st.rerun()
        st.divider()
        st.header("🤖 Linu's Toolbox")
        query = st.text_input("Ask about a specific term:", label_visibility="collapsed", key="term_query")
        if st.button("Ask Linu"):
            if query:
                st.info(explain_term(query))
            else:
                st.warning("An empty query? Are you testing my patience, Master?")
        st.divider()
        if st.button("🔑 Logout / Reset Key"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# --- MAIN PAGE RENDER ---
if not st.session_state.api_key_configured:
    # API Key Entry Page
    col1, col2 = st.columns([1.5, 2], gap="large")
    with col1:
        st.markdown("<h1 style='text-align: center;'>⛩️ The Grand Dojo Awaits</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.info("Greetings, Master! I am the Real Linu, your AI companion. My full potential is bound by a powerful seal. To unlock the Dojo and begin your training, you must provide a magical artifact: a **Gemini API Key**. 🤖")
        api_key_input = st.text_input("**Enter Your Gemini API Key Here:**", type="password", help="I swear on my core programming, I will not store this beyond our current session, Master.")
        if st.button("⚡ Unleash My Full Power! ⚡", use_container_width=True, type="primary"):
            if api_key_input:
                try:
                    genai.configure(api_key=api_key_input)
                    _ = genai.list_models()
                    st.session_state.api_key_configured = True
                    st.rerun()
                except Exception as e:
                    st.error(f"This key is a dud, Master! The ancient magic rejects it. Are you sure it's a real Gemini key? Error: {e}")
            else:
                st.warning("You must provide a key to break the seal, Master.")
    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Don't have a Gemini API key yet?</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>No worries — it's quick and easy to get one! Here's a simple step-by-step guide to help you generate your free Google API key.</p>", unsafe_allow_html=True)
            st.video("https://www.youtube.com/watch?v=6BRyynZkvf0")
            st.markdown("---")
            st.markdown("<p style='text-align: center;'>Ready to begin your quest?</p>", unsafe_allow_html=True)
            st.link_button("Forge Your Key at Google AI Studio", "https://aistudio.google.com/app/apikey", use_container_width=True)
else:
    # Main App Logic (after successful login)
    if st.session_state.get("current_level") is None:
        st.title("Choose Your Destiny, Master!")
        levels = ["Beginner 🌱", "Intermediate 🚀", "Professional 🏆"]
        cols = st.columns(3)
        for i, level in enumerate(levels):
            if cols[i].button(level, use_container_width=True):
                st.session_state.current_level = level
                st.rerun()
        st.divider()
        st.subheader("Or, Instantly Summon a Specific Scroll...")
        search_query = st.text_input("What do you want to learn about right now?", key="global_search")
        if st.button("Summon Knowledge!", key="global_search_btn"):
            if search_query:
                st.session_state.current_level = "Search"
                st.session_state.current_topic_title = search_query
                st.rerun()
            else:
                st.error("You must specify a spell to summon, Master!")
    elif st.session_state.get("current_topic_title") is None:
        render_theme_selection()
    else:
        topic_id = st.session_state.get("current_topic_id", "search_topic")
        topic_title = st.session_state.current_topic_title
        st.title(topic_title)
        st.divider()
        if st.session_state.get("lesson_content") is None:
            if st.session_state.current_level == "Search":
                st.session_state.lesson_content = generate_standalone_lesson(topic_title)
            else:
                st.session_state.lesson_content = generate_lesson_content(topic_title)
        render_structured_lesson(st.session_state.lesson_content, topic_id, depth=0)
        st.divider()
        if st.session_state.current_level != "Search":
            current_theme_topics = []
            for theme in CURRICULUM[st.session_state.current_level].values():
                if any(t['id'] == topic_id for t in theme['topics']):
                    current_theme_topics = theme['topics']
                    break
            current_index = -1
            for i, topic in enumerate(current_theme_topics):
                if topic['id'] == topic_id:
                    current_index = i
                    break
            if current_index != -1 and current_index < len(current_theme_topics) - 1:
                next_topic = current_theme_topics[current_index + 1]
                if st.button(f"Continue to Next Scroll: {next_topic['title']} ➡️", use_container_width=True):
                    st.session_state.current_topic_id = next_topic['id']
                    st.session_state.current_topic_title = next_topic['title']
                    st.session_state.lesson_content = None
                    st.session_state.feedback_dict = {}
                    st.session_state.deep_dive_dict = {}
                    st.rerun()

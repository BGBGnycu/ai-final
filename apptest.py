import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import os
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import uuid
import streamlit as st
import io
from googletrans import Translator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

translator = Translator()

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks

class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(openAI_key, prompt, model="gpt-3.5-turbo"):
    openai.api_key = openAI_key
    temperature=0.7
    max_tokens=256
    top_p=1
    frequency_penalty=0
    presence_penalty=0

    if model == "text-davinci-003":
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        message = completions.choices[0].text
    else:
        message = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Here is some initial assistant message."},
                {"role": "user", "content": prompt}
            ],
            temperature=.3,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ).choices[0].message['content']
    return message

def translate_text(text, src='en', dest='zh-tw'):
    translation = translator.translate(text, src=src, dest=dest)
    return translation.text

def generate_summary(openAI_key, file_path, start_page=1, end_page=None):
    texts = pdf_to_text(file_path, start_page=start_page, end_page=end_page)
    prompt = "Summarize the following text:\n" + " ".join(texts)
    summary = generate_text(openAI_key, prompt)
    return summary

def generate_answer_with_translation(question, openAI_key, model):
    topn_chunks = recommender(question)
    prompt = 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using (from page no.) notation. "\
              "Only answer what is asked. The answer should be short and concise. \n\nQuery: "
    
    prompt += f"{question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, model)
    
    translated_answer = translate_text(answer)
    
    return answer, translated_answer

def question_answer(file, question, openAI_key, model):
    global pdf_loaded, file_path
    
    try:
        if openAI_key.strip() == '':
            st.session_state.chat_history.append(["system", '[ERROR]: Please enter your Open AI Key.'])
            return
        if file is None and not pdf_loaded:
            st.session_state.chat_history.append(["system", '[ERROR]: Please load a PDF file.'])
            return
        if model is None or model == '':
            st.session_state.chat_history.append(["system", '[ERROR]: Please choose a model.'])
            return

        if not pdf_loaded:
            old_file_name = file.name
            file_name = os.path.join(os.path.dirname(old_file_name), str(uuid.uuid4()) + os.path.splitext(old_file_name)[-1])
            with open(file_name, 'wb') as f:
                f.write(file.getbuffer())
            file_path = file_name

            load_recommender(file_path)
            pdf_loaded = True
            
            # Generate and add summary to chat history if not already present
            if not any(msg[0] == "summary" for msg in st.session_state.chat_history):
                summary = generate_summary(openAI_key, file_path)
                st.session_state.chat_history.append(["summary", summary])

        if question.strip() == '':
            st.session_state.chat_history.append(["system", '[ERROR]: Question field is empty.'])
            return

        answer, translated_answer = generate_answer_with_translation(question, openAI_key, model)
        st.session_state.chat_history.append(["user", question])
        st.session_state.chat_history.append(["assistant", answer])
        st.session_state.chat_history.append(["assistant_translation", translated_answer])
    except Exception as e:
        st.session_state.chat_history.append(["system", f'[ERROR]: An unexpected error occurred - {str(e)}'])

def generate_text_text_davinci_003(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message

def generate_answer_text_davinci_003(question,openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using (from page no.) notation. "\
              "Only answer what is asked. The answer should be short and concise. \n\nQuery: {question}\nAnswer: "
    
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text_text_davinci_003(openAI_key, prompt,"text-davinci-003")
    return answer

def export_chat_history():
    chat_history = st.session_state.chat_history
    if not chat_history:
        return ""
    output = io.StringIO()
    for message in chat_history:
        output.write(f"{message[0].capitalize()}: {message[1].replace('Assistant:', 'Chat PDF:')}\n\n")
    return output.getvalue()

def advanced_search(texts, keywords, page_range):
    total_pages = len(texts)
    
    if page_range[0] > total_pages:
        return "Page range exceeds the total number of pages in the document."
    
    results = []
    for idx, text in enumerate(texts):
        if idx + 1 >= page_range[0] and idx + 1 <= page_range[1]:
            if any(keyword.lower() in text.lower() for keyword in keywords.split()):
                highlighted_text = text
                for keyword in keywords.split():
                    highlighted_text = re.sub(f"(?i)({keyword})", r"<span style='background-color: yellow'>\1</span>", highlighted_text)
                results.append((idx + 1, highlighted_text))
    return results

recommender = SemanticSearch()
pdf_loaded = False
file_path = None

# Streamlit app
st.title('Chat PDF')
st.markdown(""" Chat PDF allows you to ask questions about a PDF document and get answers from the document.""")

# Sidebar inputs
st.sidebar.title("Chat PDF Configuration")
openAI_key = st.sidebar.text_input('Enter your OpenAI API key here')
file = st.sidebar.file_uploader('Upload your PDF here', type=['pdf'])
question = st.sidebar.text_input('Enter your question here')
model = st.sidebar.selectbox('Select Model', ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'text-davinci-003', 'gpt-4', 'gpt-4-32k'])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button('Submit'):
    question_answer(file, question, openAI_key, model)
        

if st.sidebar.button('Export Chat History'):
    chat_history_str = export_chat_history()
    st.download_button(label='Download Chat History', data=chat_history_str, file_name='chat_history.txt')

search_keywords = st.sidebar.text_input('Enter keywords for advanced search')
search_page_range = st.sidebar.slider('Select page range for Advanced Search', 1, 10, (1, 5))

if st.sidebar.button('Advanced Search'):
    if file is None and not pdf_loaded:
        st.session_state.chat_history.append(["system", '[ERROR]: Please load a PDF file.'])
    else:
        if not pdf_loaded:
            old_file_name = file.name
            file_name = os.path.join(os.path.dirname(old_file_name), str(uuid.uuid4()) + os.path.splitext(old_file_name)[-1])
            with open(file_name, 'wb') as f:
                f.write(file.getbuffer())
            file_path = file_name
            load_recommender(file_path)
            pdf_loaded = True

        texts = pdf_to_text(file_path)
        results = advanced_search(texts, search_keywords, search_page_range)
        if results:
            st.session_state.chat_history.append(["system", 'Advanced Search Results:'])
            for page, result in results:
                st.session_state.chat_history.append(["assistant", f'Page {page}: {result}'])
        else:
            st.session_state.chat_history.append(["system", 'No results found for the given search criteria.'])
            st.markdown(f'<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 20px;"><strong>{chat[0].capitalize()}:</strong> {chat[1]}</div>', unsafe_allow_html=True)

if 'bookmarks' in st.session_state and st.session_state.bookmarks:
    st.sidebar.markdown('## Bookmarked Sections')
    for bookmark in st.session_state.bookmarks:
        st.sidebar.markdown(f'<div style="background-color: #ffff99; padding: 10px; border-radius: 5px; margin-bottom: 10px;">{bookmark}</div>', unsafe_allow_html=True)
        
start_page = st.sidebar.number_input('Start Page for Summarize Section', min_value=1, value=1)
end_page = st.sidebar.number_input('End Page for Summarize', min_value=1, value=1)

if st.sidebar.button('Summarize Section'):
    if file is None:
        st.session_state.chat_history.append(["system", '[ERROR]: Please load a PDF file.'])
    else:
        old_file_name = file.name
        file_name = os.path.join(os.path.dirname(old_file_name), str(uuid.uuid4()) + os.path.splitext(old_file_name)[-1])
        with open(file_name, 'wb') as f:
            f.write(file.getbuffer())
        file_path = file_name
        summary = generate_summary(openAI_key, file_path, start_page=start_page, end_page=end_page)
        st.session_state.chat_history.append(["summary", summary])

# 自定義 CSS 樣式
st.markdown("""
<style>
.user-box {
    background-color: #eaffea;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;  /* 增加空間 */
    text-align: right;
    display: inline-block;
    max-width: 70%;
    word-wrap: break-word;
    float: right;
}
.chat-pdf-box {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #000000;
    margin-bottom: 20px;  /* 增加空間 */
    text-align: left;
    display: inline-block;
    max-width: 70%;
    word-wrap: break-word;
    clear: both;
}
</style>
""", unsafe_allow_html=True)

# Display chat history and bookmarking functionality
if 'chat_history' in st.session_state:
    for i, chat in enumerate(st.session_state.chat_history):
        if chat[0] == "user":
            st.markdown(f'<div class="user-box"><strong>User:</strong> {chat[1]}</div>', unsafe_allow_html=True)
        elif chat[0] == "assistant":
            st.markdown(f'<div class="chat-pdf-box"><strong>Chat PDF:</strong> {chat[1]}</div>', unsafe_allow_html=True)
            if st.button('Bookmark', key=f'bookmark_{i}'):
                if 'bookmarks' not in st.session_state:
                    st.session_state.bookmarks = []
                st.session_state.bookmarks.append(chat[1])
        elif chat[0] == "assistant_translation":
            st.markdown(f'<div class="chat-pdf-box"><strong>Chat PDF (Chinese):</strong> {chat[1]}</div>', unsafe_allow_html=True)
            if st.button('Bookmark', key=f'bookmark_{i}_translation'):
                if 'bookmarks' not in st.session_state:
                    st.session_state.bookmarks = []
                st.session_state.bookmarks.append(chat[1])
        elif chat[0] == "summary":
            st.markdown(f'<div style="margin-bottom: 20px;"><strong>Summary:</strong> {chat[1]}</div>', unsafe_allow_html=True)
            if st.button('Bookmark', key=f'bookmark_{i}'):
                if 'bookmarks' not in st.session_state:
                    st.session_state.bookmarks = []
                st.session_state.bookmarks.append(chat[1])
        else:
            st.markdown(f'<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 20px;"><strong>{chat[0].capitalize()}:</strong> {chat[1]}</div>', unsafe_allow_html=True)

if 'chat_history' in st.session_state and any(chat[0] == "user" for chat in st.session_state.chat_history):
    if st.button('Generate New Answer'):
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            last_user_question = st.session_state.chat_history[-3][1]
            question_answer(file,last_user_question,openAI_key, model)
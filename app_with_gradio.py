
import os
from dotenv import load_dotenv
import queue
import tempfile
import shutil
import streamlit as st
from openai import OpenAI
from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import StreamingStdOutCallbackHandlerYield
from embedchain.models.data_type import DataType
from embedchain.loaders.base_loader import BaseLoader


# Load environment variables
load_dotenv()

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "docx", "csv", "jpeg", "jpg", "webp"]

# Retrieve OpenAI API Key from environment
OpenAI_Api_Key = os.getenv("OpenAI_Api_Key")
print(f'OpenAI_Api_Key: {OpenAI_Api_Key}')

# Function to clear the Chroma database directory
def clear_chroma_db():
    chroma_db_path = os.path.join(tempfile.gettempdir(), "chroma_db")
    shutil.rmtree(chroma_db_path, ignore_errors=True)
    print(f"Cleared Chroma database directory: {chroma_db_path}")

# Clear Chroma database directory
clear_chroma_db()

class CustomImageLoader(BaseLoader):
    def load(self, file_path: str):
        if file_path.startswith('file://'):
            file_path = file_path[7:]  # Remove 'file://' prefix
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        print(f"Loading image from: {file_path}")
        
        # Return the data in the format expected by the chunker
        return {
            "data": [
                {
                    "content": f"Image file: {os.path.basename(file_path)}",
                    "meta": {"source": file_path, "type": "image"}
                }
            ]
        }

# Initialize the Embedchain bot with specified API provider and key.
def embedchain_bot(db_path, api_key, api_provider):
    if api_provider == "OpenAI":  # OpenAI API
        llm_config = {
            "provider": "openai",
            "config": {
                "model": "gpt-4-turbo",  # Use a supported model
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 1,
                "stream": True,
            },
        }
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": 'text-embedding-ada-002',  # Updated embedding model
                "api_key": api_key
            }
        }
    else:  # Google API
        llm_config = {
            "provider": "google",
            "config": {
                "model": "gemini-pro",
                "max_tokens": 1000,
                "temperature": 0.2,
                "top_p": 1,
                "stream": False,
                "api_key": api_key
            },
        }
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": 'text-embedding-ada-002',  # Updated embedding model
                "api_key": OpenAI_Api_Key
            }
        }

    app = App.from_config(
        config={
            "llm": llm_config,
            "vectordb": {
                "provider": "chroma",
                "config": {"collection_name": "chat-files"},
            },
            "embedder": embedder_config,
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )
    return app

# Create a temporary directory for storing the database.
def get_db_path():
    tmpdirname = tempfile.mkdtemp()
    return tmpdirname

# Retrieve the Embedchain app from the session state or create a new one.
def get_ec_app(api_key, api_provider):
    if "app" in st.session_state:
        print("Found app in session state")
        app = st.session_state.app
    else:
        print("Creating app")
        db_path = get_db_path()
        app = embedchain_bot(db_path, api_key, api_provider)
    st.session_state.app = app
    return app

def custom_datatypes(filetype: str):
    if isinstance(filetype, str):
        if filetype in ['jpg', 'jpeg', 'webp']:
            return DataType.CUSTOM
        elif filetype in ['pdf']:
            return 'pdf_file'
        elif filetype in ['csv']:
            return 'csv'
        elif filetype in ['docx']:
            return 'docx'
        else:
            raise ValueError(f"Filetype not supported: '{filetype}'")
    else:
        raise ValueError(f"Filetype not a string: '{filetype}'")

# Process the uploaded file and add it to the knowledge base.
def process_file(file, app):
    try:
        file_name = file.name
        file_type = file.name.split(".")[-1]
        temp_dir = tempfile.gettempdir()
        legit_file_type = custom_datatypes(file_type)
        temp_file_path = os.path.join(temp_dir, file_name)
        
        with open(temp_file_path, 'wb') as f:
            f.write(file.getvalue())
        
        print(f"Processing file: {file_name}")
        print(f"File type: {file_type}")
        print(f"Legit file type: {legit_file_type}")
        print(f"Temp file path: {temp_file_path}")
        
        file_url = f"file://{temp_file_path}"
        
        if legit_file_type == DataType.CUSTOM:
            loader = CustomImageLoader()
            # For image files, we'll add them directly without chunking
            data = loader.load(file_url)
            result = app.add(data["data"][0]["content"], data_type=DataType.TEXT, metadata=data["data"][0]["meta"])
        else:
            result = app.add(file_url, data_type=legit_file_type)
        
        print(f"Add result: {result}")
        
        if result is None:
            raise ValueError(f"Failed to add file {file_name}. app.add() returned None.")
        
        os.remove(temp_file_path)
        st.markdown(f"Added {file_name} to knowledge base!")
        return True
    except Exception as e:
        print(f"Error in process_file for {file_name}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise  # Re-raise the exception to be caught in the main loop

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure api_key is in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Sidebar for settings
with st.sidebar:
    api_provider = st.radio("Choose API Provider", ('OpenAI', 'Google'))
    if api_provider == 'OpenAI':
        api_key = st.text_input("OpenAI API Key", key="api_key", type="password")
        "WE DO NOT STORE YOUR OPENAI KEY."
        "Just paste your OpenAI API key here and we'll use it to power the chatbot."
    else:
        api_key = st.text_input("Google API Key", key="api_key", type="password")
        "WE DO NOT STORE YOUR GOOGLE API KEY."
        "Just paste your Google API key here and we'll use it to power the chatbot."

    if st.session_state.api_key:
        os.environ["CHROMA_ENDPOINT"] = os.getenv("CHROMA_ENDPOINT")
        os.environ["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")
        print(f'CHROMA_ENDPOINT: {os.getenv("CHROMA_ENDPOINT")}')
        print(f'CHROMA_API_KEY: {os.getenv("CHROMA_API_KEY")}')
        if api_provider == 'OpenAI':
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        else:
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
        print(f'api_provider: {api_provider} --- api_key: {st.session_state.api_key}')
        app = get_ec_app(st.session_state.api_key, api_provider)

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=SUPPORTED_FILE_TYPES)
    add_files = st.session_state.get("add_files", [])
    
    for file in uploaded_files:
        print(f'Uploaded File: {file.name}')
        file_name = file.name
        if file_name in add_files:
            continue
        try:
            if not st.session_state.api_key:
                st.error("Please enter your API Key")
                st.stop()
            else:
                process_file(file, app)
            add_files.append(file_name)
        except Exception as e:
            error_message = f"Error adding {file_name} to knowledge base: {str(e)}"
            st.error(error_message)
            print(error_message)
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            # Don't stop here, continue with the next file
            continue

    st.session_state["add_files"] = add_files

# Main page
cols = st.columns([1, 2, 1])
with cols[1]:
    st.image('./logo.jpeg', use_column_width=True)
st.title("DRASCUS")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm a multi-modal chatbot, which can answer questions about your documents and images.\n
                Upload your documents here and I'll answer your questions about them! 
            """,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if (prompt := st.chat_input("Ask me anything!")):
    if not st.session_state.api_key:
        st.error("Please enter your API Key")
        st.stop()

    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")
        full_response = ""

        q = queue.Queue()

        def app_response(prompt):
            llm_config = app.llm.config.as_dict()
            
            # Remove unexpected keys
            unexpected_keys = ['http_client', 'http_async_client']
            for key in unexpected_keys:
                llm_config.pop(key, None)
            
            llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
            config = BaseLlmConfig(**llm_config)
            answer, citations = app.chat(prompt, config=config, citations=True)
            result = {}
            result["answer"] = answer
            result["citations"] = citations
            return result
        
        results = app_response(prompt)
        answer, citations = results["answer"], results["citations"]
        full_response = answer

        msg_placeholder.markdown(results['answer'])
        
        if citations:
            full_response += "\n\n**Sources**:\n"
            sources = []
            for i, citation in enumerate(citations):
                source = citation[1]["url"]
                sources.append(os.path.basename(source))
            sources = list(set(sources))
            for source in sources:
                full_response += f"- {source}\n"

        msg_placeholder.markdown(full_response)
        print("Answer: ", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        del prompt


# Gradio Import
import gradio as gr
import threading
from pyngrok import ngrok

# Define a function for image generation (Placeholder function, replace with your actual logic)
def generate_image(prompt):
    # Replace this with actual image generation logic
    return "https://via.placeholder.com/512?text=Generated+Image"

# Create a Gradio Interface
interface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    live=True,
)

# Function to run Gradio in a separate thread
def run_gradio():
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Start Gradio in a separate thread
thread = threading.Thread(target=run_gradio)
thread.start()

# Use ngrok to create a public URL for the Gradio app
public_url = ngrok.connect(7860).public_url
st.markdown(f'<iframe src="{public_url}" width="100%" height="500px"></iframe>', unsafe_allow_html=True)

st.title("Multimodal RAG Application with Image Generation")

# The rest of your Streamlit code...

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

 # Set the page configuration with a white background
st.set_page_config(
    page_title="KOgenie",
    page_icon="icon.png",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Apply custom CSS to set the background color to white
st.markdown(
    """
    <style>

    .main {
        background-color: #E7DECC;
    }
    

    .st-emotion-cache-usj992 {
    position: fixed;
    bottom: 0px;
    padding-bottom: 70px;
    padding-top: 1rem;
    background-color:#E7DECC ;
    z-index: 99;
}
.st-emotion-cache-10trblm {
    position: relative;
    flex: 1 1 0%;
    margin-left: calc(3rem);
    color: black;
}
.st-d5{
    cursor:pointer;
}
p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    color: white;
}
    </style>
    """,
    unsafe_allow_html=True
)


# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "docx", "csv", "jpeg", "jpg", "webp", "png"]

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
        
        return {
            "data": [
                {
                    "content": f"Image file: {os.path.basename(file_path)}",
                    "meta": {"source": file_path, "type": "image"}
                }
            ]
        }

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

def get_db_path():
    tmpdirname = tempfile.mkdtemp()
    return tmpdirname

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
            data = loader.load(file_url)
            result = app.add(data["data"][0]["content"], data_type=DataType.TEXT, metadata=data["data"][0]["meta"])
        else:
            result = app.add(file_url, data_type=legit_file_type)
        
        print(f"Add result: {result}")
        
        if result is None:
            raise ValueError(f"Failed to add file {file_name}. app.add() returned None.")
        
        os.remove(temp_file_path)
        st.success(f"Successfully added **{file_name}** to knowledge base!")
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
    st.title("KOgenie Settings")
    api_provider = st.selectbox("Choose API Provider", ('OpenAI', 'Google'))
    if api_provider == 'OpenAI':
        api_key = st.text_input("OpenAI API Key", key="api_key", type="password")
        st.caption("We do not store your OpenAI key.")
    else:
        api_key = st.text_input("Google API Key", key="api_key", type="password")
        st.caption("We do not store your Google API key.")

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
            st.error(f"Error adding **{file_name}** to knowledge base: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            continue

    st.session_state["add_files"] = add_files

# Main page
st.image('./logo-2.svg', use_column_width=True)
st.title("KOgenie")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm KOgenie, a multi-modal chatbot here to answer your questions about documents and images.\n
                Upload your documents, and I'll assist you with any queries!
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


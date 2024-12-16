import streamlit as st
import os
import tempfile
import pdfplumber
from PIL import Image
import docx
import pandas as pd
from dotenv import load_dotenv
import queue
import openai
import subprocess
from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import StreamingStdOutCallbackHandlerYield
from embedchain.models.data_type import DataType
from embedchain.loaders.base_loader import BaseLoader

# Load environment variables
load_dotenv()

# Retrieve OpenAI API Key from environment
OpenAI_Api_Key = os.getenv("OPENAI_API_KEY")
if OpenAI_Api_Key is None:
    st.error("OpenAI API Key not found in environment variables.")
    st.stop()

print(f'OPENAI_API_KEY: {OpenAI_Api_Key}')

# Correct import for OpenAI API
client = openai

# Initialize the Embedchain bot with specified API provider and key.
def embedchain_bot(api_key):
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

    app = App.from_config(
        config={
            "llm": llm_config,
            "embedder": embedder_config,
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )
    return app

# Custom image loader for handling image files
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

# Function to generate images using DALL-E in a subprocess with OpenAI version 0.28
def generate_image_dalle(prompt):
    try:
        # Replace newline characters with spaces to prevent syntax errors
        sanitized_prompt = prompt.replace("\n", " ")

        result = subprocess.run(
            [
                "image_env/bin/python",  # Use the Python interpreter from the image_env
                "-c",
                f"""
import openai
openai.api_key = '{OpenAI_Api_Key}'
response = openai.Image.create(
    prompt='{sanitized_prompt}',
    n=1,
    size="1024x1024"
)
print(response['data'][0]['url'])
"""
            ],
            capture_output=True,
            text=True
        )

        # Print stdout and stderr for debugging purposes
        print("Subprocess STDOUT:", result.stdout)
        print("Subprocess STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed with return code {result.returncode}")

        return result.stdout.strip()
    except Exception as e:
        print(f"Error generating image: {e}")
        st.error(f"Failed to generate image: {str(e)}")
        return None


# Ensure api_key is in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = OpenAI_Api_Key  # Initialize with the environment variable if available

# Ensure api_key is in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = OpenAI_Api_Key  # Initialize with the environment variable if available

# Initialize the Embedchain bot with the API key
app = embedchain_bot(st.session_state.api_key)

# Main page for interaction
cols = st.columns([1, 2, 1])
with cols[1]:
    st.image('./logo-2.svg', use_column_width=True)
st.title("KOgenie")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm a multi-modal chatbot. Create an advertisement for a new product and I'll generate an image for it!\n 
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

        # Generate image using DALL-E
        image_url = generate_image_dalle(answer)
        if image_url:
            st.image(image_url, caption="Generated Image", use_column_width=True)

        msg_placeholder.markdown(full_response)

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
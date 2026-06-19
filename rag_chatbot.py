from dotenv import load_dotenv
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_dfdnvugyfchsu-uffdybf"
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# optional decorator to print line by line
# import pprint


load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Step 1a - Indexing (Document Ingestion)
# video_id = "k4yz7ZP9GA4"  # 18
# video_id = "2HIJlD46Kig" # 91
video_id = "J5_-l7WIO_w"  # 51

try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en", "hi"])

    # Convert to raw list of dicts and join text
    transcript = " ".join(chunk["text"] for chunk in fetched_transcript.to_raw_data())
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video")
except Exception as e:
    print(f"Error: {e}")

# Step 1b Indexing Text-spiliting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))

print(chunks[50].page_content)

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

faiss_path = "faiss_db"

if os.path.exists(faiss_path):
    # Load existing vector store — IDs stay the same
    vector_store = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded existing vector store")
else:
    # Build new vector store only once
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(faiss_path)
    print("Created new vector store")

print(vector_store.index_to_docstore_id)

# print(vector_store.get_by_ids(["93457261-45d5-462c-97a2-364c9e5d18c0"]))

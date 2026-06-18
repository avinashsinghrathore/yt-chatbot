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

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Step 1a - Indexing (Document Ingestion)
# video_id = "dQw4w9WgXcQ"  # only the id not full url
# try:
#     transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=["en","hi"])

#     transcript = " ".join(chunk["text"] for chunk in transcript_list)
#     print(transcript)
# except TranscriptsDisabled:
#     print("No caption available for this video")


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

print(chunks[0].page_content)
print(chunks[0].metadata)

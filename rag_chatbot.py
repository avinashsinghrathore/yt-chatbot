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
# video_id = "2HIJlD46Kig"  # 91
# video_id = "J5_-l7WIO_w"  # 51
video_id = "-HzgcbRXUK8"  # 184

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
# print(transcript)
print(len(chunks))
# print(chunks[50].page_content)

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

# print(vector_store.get_by_ids(["50d483ea-e4f0-4b42-b818-a666d8fc0a32"]))

# Step 2 Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriver)

# result = retriever.invoke("what is deepmind")
# print(result)

# Step 3 Augmentation
# model setup
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", task="conversational", temperature=0.2
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template=""" You are helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know  
    {context}
    Question: {question}                                                         
    """,
    input_variables=["context", "question"],
)

question = "is the topic of nuclear discussed in this video ? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

# print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

# Generation
answer = model.invoke(final_prompt)
print(answer.content)

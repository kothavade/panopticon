import io
import os
import sys
import logging
from enum import Enum
from typing import Any

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec
from yt_dlp import YoutubeDL

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

URLs = [
    "https://umd.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=df62a33f-4d8b-438d-bca9-b14000e1b249"
]

LECTURES_DIR = "lectures"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def setup():
    if not os.path.exists("lectures"):
        logging.info("Creating lectures directory")
        os.mkdir("lectures")
    # TODO: figure out setup for whisper-cpp
    # if not os.path.exists("models"):
    #     os.mkdir("lectures")
    # if "ggml-base.en.bin" not in os.listdir("models"):
    #     os.system("whisper-cpp-download-ggml-model base.en")
    #     os.system("mv ggml-base.en.bin models/ggml-base.en.bin")


class Stage(Enum):
    # value corresponds to expeceted file extension for the stage
    DOWNLOAD = "wav"
    TRANSCRIBE = "transcript"
    SUMMARIZE = "summary"


def filter_urls(URLs: list[str], stage: Stage) -> list[str]:
    filtered: list[str] = []
    for URL in URLs:
        id = URL.split("id=")[1]
        if f"{id}.{stage.value}" not in os.listdir(LECTURES_DIR):
            filtered.append(URL)
        else:
            logging.info(f"Skipping {id} as {stage.value} exists")
    return filtered


def get_opts(cookies: str) -> dict[str, Any]:
    return {
        "cookiefile": io.StringIO(cookies),
        "format": "worst",  # audio seems unaffected by this
        "outtmpl": f"{LECTURES_DIR}/%(id)s.%(ext)s",
        # whisper-cpp requires 16kHz audio
        "postprocessor_args": {"ffmpeg": ["-ar", "16000"]},
        # extract audio as wav
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }


def main():
    # get urls (eventually, this comes from the frontend)
    urls = sys.argv[1:]
    if urls:
        URLs.extend(urls)

    # get cookies (eventually, this comes from the frontend)
    with open("cookies.txt", "r") as f:
        cookies = f.read()

    opts = get_opts(cookies)

    # if transcript exists, skip download
    filtered = filter_urls(URLs, Stage.TRANSCRIBE)
    # if download exists, skip download
    filtered = filter_urls(filtered, Stage.DOWNLOAD)

    # download
    with YoutubeDL(opts) as ydl:
        _ = ydl.download(filtered)

    logging.info("Finished downloading")

    # transrcibe
    for URL in filtered:
        id = URL.split("id=")[1]
        cmd = f"whisper-cpp -osrt -otxt -m models/tiny.en.bin {LECTURES_DIR}/{id}.wav"
        _ = os.system(cmd)
        _ = os.system(f"mv {LECTURES_DIR}/{id}.wav.srt {LECTURES_DIR}/{id}.srt")
        _ = os.system(f"mv {LECTURES_DIR}/{id}.wav.txt {LECTURES_DIR}/{id}.transcript")

    logging.info("Finished transcribing")

    # summarize
    filtered = filter_urls(URLs, Stage.SUMMARIZE)

    # configure llama_index
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # logging.info("Deleting index")
    # pc.delete_index("panopticon")
    # logging.info("Creating index")
    # pc.create_index(
    #     name="panopticon",
    #     dimension=768,
    #     metric="cosine",
    #     spec=PodSpec(environment="gcp-starter"),
    # )
    pinecone_index = pc.Index("panopticon")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = GeminiEmbedding(api_key=GEMINI_API_KEY)
    Settings.llm = Gemini(api_key=GEMINI_API_KEY)

    logging.info("Loading documents")
    documents = SimpleDirectoryReader(
        LECTURES_DIR, required_exts=[".transcript"]
    ).load_data()
    logging.info("Creating index from documents")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logging.info("Querying index")
    query_engine = index.as_query_engine()
    while True:
        query = input("Enter query: ")
        if query == "exit":
            break
        response = query_engine.query(query)
        print(response)
        print()


if __name__ == "__main__":
    setup()
    main()

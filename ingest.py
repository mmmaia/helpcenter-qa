import re
import requests
from bs4 import BeautifulSoup

import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


def get_html(url):
    response = requests.get(url)
    return response.text


def get_section_links(url, section):
    html_doc = get_html(url)
    soup = BeautifulSoup(html_doc, "html.parser")

    return [ "https://ajuda.infinitepay.io" + link.get('href') for link in soup.find_all('a', attrs={ 'href': re.compile(f"/{section}/") }) ]


def get_intercom_links(url):
    intercom_urls = []
    for collection_url in get_section_links(url, 'collections'):
        links = get_section_links(collection_url, 'articles')
        intercom_urls.extend(links)
    
    return intercom_urls


def get_clean_data(url):
    html_doc = get_html(url)
    soup = BeautifulSoup(html_doc, "html.parser")
    text = soup.find("article")
    for match in text.findAll('p'):
        match.extend("\n")
    text = text.get_text().strip()
    return "\n".join([t for t in text.split("\n") if t])


links = get_intercom_links("https://ajuda.infinitepay.io/pt-BR/")

texts = []
metadatas = []
for link in links:
    print(link)
    # print(get_clean_data(link))
    texts.append( get_clean_data(link) )
    metadatas.append( {"source": link} )


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)
documents = text_splitter.create_documents(texts, metadatas=metadatas)

# QA 2
with open("search_index.pickle", "wb") as f:
    pickle.dump(FAISS.from_documents(documents, OpenAIEmbeddings()), f)

# QA
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)
documents = []
metadatas = []
for i, d in enumerate(texts):
    splits = text_splitter.split_text(d)
    documents.extend(splits)
    metadatas.extend([{"source": links[i]}] * len(splits))

store = FAISS.from_texts(documents, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
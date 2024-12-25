# Building a RAG Chatbot with LangChain

This project demonstrates how to build a chatbot capable of learning from external knowledge sources using Retrieval Augmented Generation (RAG). The chatbot utilizes LangChain, OpenAI, and Pinecone to enable dynamic responses informed by a custom knowledge base. By the end of this project, we will have a functioning chatbot capable of answering questions about the latest advancements in Generative AI (GenAI) using a dataset sourced from Llama 2 ArXiv papers.

---

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Steps](#steps)
    - [Building a Chatbot (No RAG)](#building-a-chatbot-no-rag)
    - [Importing the Data](#importing-the-data)
    - [Building the Knowledge Base](#building-the-knowledge-base)
    - [Integrating RAG](#integrating-rag)
5. [Usage](#usage)
6. [Results](#results)
7. [Cleaning Up](#cleaning-up)
8. [Acknowledgements](#acknowledgements)

---

## Overview
LLMs like OpenAI's models have a fixed knowledge base learned during training, which limits their ability to provide up-to-date or specific responses. To address this limitation, this project:
- Leverages Retrieval Augmented Generation (RAG) to integrate external data sources.
- Uses a vector database (Pinecone) to store and retrieve knowledge chunks.
- Processes queries with the retrieved context to provide informed and accurate responses.

---

## Requirements

### API Keys
- [OpenAI API Key](https://platform.openai.com/signup/)
- [Pinecone API Key](https://www.pinecone.io/start-free/)

### Libraries
- `langchain`: For chaining together the components of the chatbot.
- `openai`: To interact with the OpenAI API.
- `datasets`: To load the knowledge base.
- `pinecone-client`: To interact with the Pinecone vector database.

Install the required libraries using the following command:
```bash
pip install langchain openai datasets pinecone-client
```

---

## Setup
1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2. Configure environment variables for API keys:
    ```bash
    export OPENAI_API_KEY=<your_openai_api_key>
    export PINECONE_API_KEY=<your_pinecone_api_key>
    export PINECONE_ENVIRONMENT=<your_pinecone_environment>
    ```

---

## Steps

### Building a Chatbot (No RAG)
- Initialize a chatbot using the `ChatOpenAI` class from LangChain.
- The chatbot processes user input but lacks external knowledge, leading to limited responses on recent topics.

```python
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.7)
messages = []
prompt = HumanMessage(content="What is Llama 2?")
response = chat(messages + [prompt])
print(response.content)
```

### Importing the Data
- Use the Hugging Face Datasets library to load a dataset of Llama 2-related ArXiv papers.
- Each dataset entry represents a chunk of text to be stored in the knowledge base.

```python
from datasets import load_dataset

dataset = load_dataset("jamescalam/llama-2-arxiv-papers")
```

### Building the Knowledge Base
- Connect to Pinecone and initialize a vector index.
- Use OpenAI's `text-embedding-ada-002` model to create 1536-dimensional vector embeddings for each dataset chunk.

```python
import pinecone
from langchain.embeddings import OpenAIEmbeddings

pinecone.init(api_key=<PINECONE_API_KEY>, environment=<PINECONE_ENVIRONMENT>)
index = pinecone.Index(<index_name>)
embeddings = OpenAIEmbeddings()
```

### Integrating RAG
- Enhance the chatbot by connecting the vector index to a LangChain `vectorstore`.
- Query the knowledge base for relevant context and append it to the chatbot prompt.

```python
from langchain.vectorstores import Pinecone

vectorstore = Pinecone(index, embeddings.embed_query, "openai")
prompt = HumanMessage(content="What safety measures were used in the development of Llama 2?")
response = chat(messages + [prompt], knowledge=vectorstore)
print(response.content)
```

---

## Usage
Run the script to start the chatbot. You can ask it questions related to Llama 2 or other topics in the dataset.

---

## Results
The chatbot provides informed responses to user queries, leveraging the RAG pipeline to include additional context and reduce hallucinations. Example:

**Query:** What safety measures were used in the development of Llama 2?

**Response:**
- Safety-specific data annotation and tuning.
- Red-teaming and iterative evaluations.
- Emphasis on responsible development and sharing methodology for reproducibility.

---

## Cleaning Up
To save resources, delete the Pinecone index when done:
```python
pinecone.delete_index(index_name)
```

---

# Edge AI

```mermaid
graph TB
    User((User))

    subgraph "FastAPI Backend"
        APIServer["API Server<br>(FastAPI)"]
        
        subgraph "Authentication Components"
            AuthManager["Auth Manager<br>(OAuth2 + JWT)"]
            UserManager["User Manager<br>(SQLAlchemy)"]
        end

        subgraph "File Processing Components"
            FileUploader["File Uploader<br>(FastAPI File)"]
            PDFProcessor["PDF Processor<br>(PyPDF)"]
            TextProcessor["Text Processor<br>(LangChain)"]
        end

        subgraph "Chat Components"
            ChatManager["Chat Manager<br>(LangChain)"]
            ConversationMemory["Conversation Memory<br>(Buffer Memory)"]
            LLMService["LLM Service<br>(Ollama)"]
        end

        subgraph "Vector Store Components"
            EmbeddingService["Embedding Service<br>(FastEmbed)"]
            VectorStoreManager["Vector Store Manager<br>(ChromaDB)"]
        end
    end

    subgraph "Data Storage"
        UserDB[("User Database<br>(SQLite)")]
        PDFVectorDB[("PDF Vector Store<br>(ChromaDB)")]
        TextVectorDB[("Text Vector Store<br>(ChromaDB)")]
        ChatHistoryDB[("Chat History Store<br>(ChromaDB)")]
    end

    subgraph "External Services"
        OllamaLLM["Ollama LLM<br>(External LLM)"]
    end

    %% User interactions
    User -->|"Authenticates"| APIServer
    User -->|"Uploads Files"| APIServer
    User -->|"Chats"| APIServer

    %% API Server connections
    APIServer -->|"Uses"| AuthManager
    APIServer -->|"Uses"| FileUploader
    APIServer -->|"Uses"| ChatManager

    %% Authentication flow
    AuthManager -->|"Manages"| UserManager
    UserManager -->|"Stores"| UserDB

    %% File processing flow
    FileUploader -->|"Processes"| PDFProcessor
    FileUploader -->|"Processes"| TextProcessor
    PDFProcessor -->|"Uses"| EmbeddingService
    TextProcessor -->|"Uses"| EmbeddingService

    %% Vector store interactions
    EmbeddingService -->|"Stores"| VectorStoreManager
    VectorStoreManager -->|"Manages"| PDFVectorDB
    VectorStoreManager -->|"Manages"| TextVectorDB
    VectorStoreManager -->|"Manages"| ChatHistoryDB

    %% Chat flow
    ChatManager -->|"Uses"| ConversationMemory
    ChatManager -->|"Uses"| LLMService
    ChatManager -->|"Retrieves from"| VectorStoreManager
    LLMService -->|"Calls"| OllamaLLM

    %% Storage relationships
    PDFProcessor -->|"Stores vectors"| PDFVectorDB
    TextProcessor -->|"Stores vectors"| TextVectorDB
    ChatManager -->|"Stores history"| ChatHistoryDB
```
## Description
This is a prototype of an edge cloud system specializing in AI, designed to empower businesses utilizing LangChain. It facilitates the creation of Retrieval Augmented Generation (RAG) AI models trained exclusively on a company's or organization's proprietary data.

## Tools used in this project
* [LangChain](https://www.langchain.com/): LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications.
* [Ollama](https://ollama.com/): Ollama is an advanced AI tool that enables users to run large language models (LLMs) locally on their personal computers.
* [FastApi](https://fastapi.tiangolo.com/): FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
* [Llama3.1](https://www.llama.com/llama3.1/): LLM model

## Project Structure

- **api/**: Contains the main FastAPI application file ([`app.py`](api/app.py)).
- **src/**: Contains modules for data processing, file uploading, and chat.
- **Dockerfile**: Defines the Docker container configuration.
- **requirements.txt**: List of project dependencies.
- **Readme.md**: Project description and how to run it.
- **.gitignore**: Files and directories to be ignored by Git.

## File Descriptions

### [`api/app.py`](api/app.py)

This file defines the FastAPI application with several endpoints:

1. **`/register/`**: Registers a new user.
2. **`/logout/`**: Logs out the current user.
3. **`/token`**: Logs in a user and returns an access token.
4. **`/users/me`**: Retrieves the current user's information.
5. **`/text/`**: Processes and stores text in a vectorstore.
6. **`/upload/`**: Uploads a PDF file, processes it, and stores it in a vectorstore.
7. **`/chat/`**: Generates a response to a message using a language model and a vectorstore.
8. **`/delete-pdf/`**: Deletes a PDF file from the vectorstore.
9. **`/get-pdfs/`**: Retrieves the list of processed PDFs for the current user.
10. **`/new-chat/`**: Creates a new chat and clears the conversation memory.
11. **`/chat-history/{chat_id}`**: Retrieves the chat history for a specific chat.
12. **`/chat-list/`**: Retrieves a list of all chats for the current user.
13. **`/load-chat/`**: Loads a past chat to continue the conversation.

### [`src/upload_files.py`](src/upload_files.py)

This module contains functions for processing PDF files, creating vectorstores, sanitizing file names, and deleting PDF files from the server and the vectorstore.

### [`src/chat.py`](src/chat.py)

This module defines the [`chat`](src/chat.py) function that generates responses to messages using a language model and a vectorstore. It uses conversation memory to maintain the context of the conversation.

### [`src/get_data.py`](src/get_data.py)

This module contains the [`process_and_store_text_data`](src/get_data.py) function that processes and stores text in a vectorstore.

## Project Functionality

1. **User Registration and Authentication**: The `/register/`, `/logout/`, `/token`, and `/users/me` endpoints handle user registration, login, logout, and retrieval of user information.

2. **Text Processing**: The `/text/` endpoint receives text and optional metadata, processes it, and stores it in a vectorstore using embeddings generated by the [`FastEmbedEmbeddings`](api/app.py) model.

3. **PDF File Upload**: The `/upload/` endpoint allows uploading PDF files. The files are processed and stored in a vectorstore. Embeddings are generated for the documents extracted from the PDF.

4. **Chat**: The `/chat/` endpoint allows users to send messages and receive responses generated by a language model. It uses an [`EnsembleRetriever`](src/chat.py) to combine text and PDF data in the responses.

5. **PDF File Deletion**: The `/delete-pdf/` endpoint allows deleting PDF files from the vectorstore and the server.

6. **Chat Management**: The `/new-chat/`, `/chat-history/{chat_id}`, `/chat-list/`, and `/load-chat/` endpoints handle creating new chats, retrieving chat history, listing all chats, and loading past chats to continue the conversation.

## Handling Multiple Users

The project handles multiple users as follows:

1. **Conversation Memory**: Uses [`ConversationBufferMemory`](src/chat.py) to maintain the context of each user's conversation. The memory can be reset if the user requests it.

2. **Separate Vectorstores**: Uses different vectorstores to store text and PDF data, allowing efficient information retrieval.

3. **Independent Endpoints**: Each endpoint is independent and can handle concurrent requests from multiple users thanks to the asynchronous nature of FastAPI.

4. **Authentication and Security**: Implements authentication and authorization to ensure that each user's data is protected and accessible only by the corresponding user.

## Running the Project

### Run the project Theta Edge Cloud

1. **Download curl**:
    ```bash
    apt install curl libcurl4-openssl-dev
    ```
2. **Download ollama**:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
3. **On ollama**:
    ```bash
    ollama serve
    ```
4. **Run project**
    ```bash
    uvicorn api.app:app
    ```
5. **First Steps**:Performs fine tuning of the model in either the pdf data or historical data endpint.
6. **Uplooad to Ollama**:After fine tuning upload the model to ollama with the endpoint upload-to-ollama
7. **Use AppAI**:You can now make queries from the AppAi with the trained model.

### Run the project with doocker

1. **Build the Docker Image**:
    ```bash
    docker build -t my-fastapi-app .
    ```

2. **Run the Docker Container**:
    ```bash
    docker run -p 5000:5000 my-fastapi-app
    ```

This will expose the application on port 5000, allowing users to interact with the defined endpoints.

In summary, this project uses FastAPI to create an API that allows processing and storing text and PDF data, generating responses to messages, and handling multiple users through conversation memory and separate vectorstores.
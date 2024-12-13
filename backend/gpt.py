import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# Define relevant paths
current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_storage_path = os.path.join(current_dir, "db/my_local_data")

# Define the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

db = Chroma(
    collection_name = "website_knowledge", 
    persist_directory = persistent_storage_path, 
    embedding_function = embedding_model)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history. Do NOT answer the question, just"
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

#Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-asnwering tasks at a property website. Use"
    "the following pieces of retrieved information to answer the user's question."
    "If you don't know the answer, just say they you don't know."
    "User three sentences or less to answer the question adn keep the answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question asnweriign
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain that combines the history-aware retriever and the QA system
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Starting a new conversation... and type 'exit' to end the conversation")  # Debugging line
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the response from llm
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()
# TODO Creating a template to greet a newcoming customer
# TODO Creating a template to answer generat quations like customer asking for a product displayed in a site
# TODO Creating a template to say goodbye to a customer
# TODO Creating a template to answer questions about the company
# TODO Creating a template to contacts the company and ask for help or put any recommendation
# TODO Creating a template to ask for a email
# TODO Creating a template to ask for a phone number



from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
import textwrap
from neo4j_env import *

class VectorRAG:
    def __init__(self):
        # Use local HuggingFace embeddings (no API quota limits)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=VECTOR_INDEX_NAME,
            node_label=VECTOR_NODE_LABEL,
            text_node_properties=[VECTOR_SOURCE_PROPERTY],
            embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
        )

        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.combine_docs_chain = create_stuff_documents_chain(ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY), self.retrieval_qa_chat_prompt)
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(),
            combine_docs_chain=self.combine_docs_chain
        )

    def query(self, question: str) -> str:
        result = self.retrieval_chain.invoke(input={"input": question})
        return textwrap.fill(result['answer'], 60)



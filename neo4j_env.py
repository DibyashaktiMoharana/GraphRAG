from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
load_dotenv('.env', override=True)
# Warning control
import warnings
warnings.filterwarnings("ignore")

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')



# Global constants
VECTOR_INDEX_NAME = 'NapoleonOpenAI'
VECTOR_NODE_LABEL = 'Napoleon_Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbeddingOpenAI'


graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
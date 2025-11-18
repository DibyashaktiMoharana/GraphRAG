"""
Script to import data into Neo4j Aura
This will create the knowledge graph structure and populate it with data
"""

from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from KnowledgeGraph.chunking import split_data_from_file
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

print(f"Connecting to Neo4j at {NEO4J_URI}...")

# Initialize Neo4j graph connection
kg = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    database=NEO4J_DATABASE
)

print("✓ Connected to Neo4j successfully!")

def create_nodes_and_relationships():
    """Create the base nodes and relationships in the knowledge graph"""
    print("\n1. Creating nodes and relationships...")
    
    cypher = """ 
    // Person Nodes
    CREATE (napoleon:Person {
        name: "Napoleon Bonaparte"
    })

    CREATE (talleyrand:Person {
        name: "Charles-Maurice de Talleyrand"
    })

    // Event node
    CREATE (waterloo:Event {
        name: "Battle of Waterloo"
    })

    // sub-event nodes
    CREATE (waterlooGeneral:General_info {
        chunk_info: "General information",
        battleDate: "1815-06-18",
        location: "Waterloo, Belgium",
        outcome: "Decisive defeat for Napoleon",
        commander: "Duke of Wellington"
    })
    CREATE (waterlooReason:Reason {
        chunk_info: "Reason",
        cause: "Napoleon's return from exile",
        strategicMistake: "Failure to unite forces before battle",
        politicalImpact: "End of the Napoleonic Wars"
    })

    CREATE (waterlooCombatant:Combatant {
        chunk_info: "Combatant",
        frenchCommander: "Napoleon Bonaparte",
        alliedCommander: "Duke of Wellington",
        prussianCommander: "Gebhard Leberecht von Blücher",
        mainForces: "French Army, British Army, Prussian Army"
    })

    CREATE (waterlooConsequence:Consequence {
        chunk_info: "Consequence",
        immediateResult: "Napoleon's second abdication",
        longTermImpact: "Restoration of the Bourbon monarchy",
        geopoliticalEffect: "Redrawing of European borders"
    })

    // Sub-person nodes: Napoleon-General-Info
    CREATE (napoleonGeneral:General_info {
        chunk_info: "General Information",
        birthDate: "1769-08-15",
        deathDate: "1821-05-05",
        nationality: "French",
        knownFor: "Military and political leader"
    })

    // Sub-person nodes: Talleyrand-General-Info
    CREATE (talleyrandGeneral:General_info {
        chunk_info: "General Information",
        birthDate: "1754-02-02",
        deathDate: "1838-05-17",
        nationality: "French",
        knownFor: "Diplomat and statesman"
    })

    // Sub-person nodes: Napoleon-Career
    CREATE (napoleonCareer:Career {
        position: "Emperor",
        period: "1804-1814",
        chunk_info: "Career"
    })

    // Sub-person nodes: Talleyrand-Career
    CREATE (talleyrandCareer:Career {
        position: "Foreign Minister",
        period: "1799-1807",
        chunk_info: "Career"
    })

    // Sub-person nodes: Napoleon-Death
    CREATE (napoleonDeath:Death {
        date: "1821-05-05",
        location: "Longwood, Saint Helena",
        chunk_info: "Death"
    })

    // Sub-person nodes: Talleyrand-Death
    CREATE (talleyrandDeath:Death {
        date: "1838-05-17",
        location: "Paris, France",
        chunk_info: "Death"
    })

    // Create relationships for career and death information
    CREATE (napoleon)-[:HAS_Career_INFO]->(napoleonCareer)
    CREATE (napoleon)-[:HAS_Death_INFO]->(napoleonDeath)
    CREATE (napoleon)-[:HAS_General_INFO]->(napoleonGeneral)

    CREATE (talleyrand)-[:HAS_Career_INFO]->(talleyrandCareer)
    CREATE (talleyrand)-[:HAS_Death_INFO]->(talleyrandDeath)
    CREATE (talleyrand)-[:HAS_General_INFO]->(talleyrandGeneral)

    // Create relationships between Person nodes
    CREATE (napoleon)-[:RELATED_TO]->(talleyrand)
    CREATE (talleyrand)-[:RELATED_TO]->(napoleon)
    
    // Create relationships between Person nodes and Event
    CREATE (napoleon)-[:RELATED_TO]->(waterloo)
    CREATE (talleyrand)-[:RELATED_TO]->(waterloo)

    // Create relationships between waterloo nodes
    CREATE (waterloo)-[:HAS_General_INFO]->(waterlooGeneral)
    CREATE (waterloo)-[:HAS_Reason_INFO]->(waterlooReason)
    CREATE (waterloo)-[:HAS_Combatant_INFO]->(waterlooCombatant)
    CREATE (waterloo)-[:HAS_Consequence_INFO]->(waterlooConsequence)
    """
    
    kg.query(cypher)
    print("✓ Nodes and relationships created!")


def import_chunks():
    """Import text chunks and create embeddings"""
    print("\n2. Importing text chunks with embeddings...")
    print("   Using local sentence-transformers model (no API needed)...")
    
    # Initialize HuggingFace embeddings (runs locally, free)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Process each JSON file
    files = [
        'data/json/Napoleon.json',
        'data/json/Talleyrand.json',
        'data/json/Battle of Waterloo.json'
    ]
    
    for file in files:
        print(f"\n  Processing {file}...")
        chunks = split_data_from_file(file)
        
        # Determine node label based on file
        if 'Napoleon' in file:
            node_label = 'Napoleon_Chunk'
        elif 'Talleyrand' in file:
            node_label = 'Talleyrand_Chunk'
        elif 'Waterloo' in file:
            node_label = 'Waterloo_Chunk'
        else:
            node_label = 'Chunk'
        
        # Create vector store from chunks
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
        
        Neo4jVector.from_texts(
            texts=texts,
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            node_label=node_label,
            text_node_property='text',
            embedding_node_property='textEmbeddingOpenAI',
            metadatas=metadatas
        )
        
        print(f"  ✓ Imported {len(chunks)} chunks as {node_label}")


def connect_chunks_to_nodes():
    """Connect text chunks to their respective nodes"""
    print("\n3. Connecting chunks to knowledge graph nodes...")
    
    queries = [
        # Napoleon Career
        """
        MATCH (napoleonCareer:Career {position: "Emperor"}), (careerChunks:Napoleon_Chunk)
        WHERE napoleonCareer.chunk_info = careerChunks.formItem 
        WITH napoleonCareer, careerChunks
        MERGE (napoleonCareer)-[r:HAS_Chunk_INFO]->(careerChunks)
        RETURN count(r) as connected
        """,
        
        # Napoleon Death
        """
        MATCH (napoleonDeath:Death {location: "Longwood, Saint Helena"}), (deathChunks:Napoleon_Chunk)
        WHERE napoleonDeath.chunk_info = deathChunks.formItem 
        WITH napoleonDeath, deathChunks
        MERGE (napoleonDeath)-[r:HAS_Chunk_INFO]->(deathChunks)
        RETURN count(r) as connected
        """,
        
        # Napoleon General Info
        """
        MATCH (napoleonGeneral:General_info {knownFor: "Military and political leader"}), (generalChunks:Napoleon_Chunk)
        WHERE napoleonGeneral.chunk_info = generalChunks.formItem
        WITH napoleonGeneral, generalChunks
        MERGE (napoleonGeneral)-[r:HAS_Chunk_INFO]->(generalChunks)
        RETURN count(r) as connected
        """,
        
        # Talleyrand Career
        """
        MATCH (TalleyrandCareer:Career {position: "Foreign Minister"}), (careerChunks:Talleyrand_Chunk)
        WHERE TalleyrandCareer.chunk_info = careerChunks.formItem 
        WITH TalleyrandCareer, careerChunks
        MERGE (TalleyrandCareer)-[r:HAS_Chunk_INFO]->(careerChunks)
        RETURN count(r) as connected
        """,
        
        # Talleyrand Death
        """
        MATCH (TalleyrandDeath:Death {location: "Paris, France"}), (careerChunks:Talleyrand_Chunk)
        WHERE TalleyrandDeath.chunk_info = careerChunks.formItem 
        WITH TalleyrandDeath, careerChunks
        MERGE (TalleyrandDeath)-[r:HAS_Chunk_INFO]->(careerChunks)
        RETURN count(r) as connected
        """,
        
        # Talleyrand General Info
        """
        MATCH (TalleyrandGeneral:General_info {knownFor: "Diplomat and statesman"}), (generalChunks:Talleyrand_Chunk)
        WHERE TalleyrandGeneral.chunk_info = generalChunks.formItem
        WITH TalleyrandGeneral, generalChunks
        MERGE (TalleyrandGeneral)-[r:HAS_Chunk_INFO]->(generalChunks)
        RETURN count(r) as connected
        """,
        
        # Waterloo General Info
        """
        MATCH (waterlooGeneral:General_info {chunk_info: "General information"}), (waterlooChunks:Waterloo_Chunk)
        WHERE waterlooGeneral.chunk_info = waterlooChunks.formItem
        WITH waterlooGeneral, waterlooChunks
        MERGE (waterlooGeneral)-[r:HAS_Chunk_INFO]->(waterlooChunks)
        RETURN count(r) as connected
        """,
        
        # Waterloo Reason
        """
        MATCH (waterlooReason:Reason {chunk_info: "Reason"}), (waterlooChunks:Waterloo_Chunk)
        WHERE waterlooReason.chunk_info = waterlooChunks.formItem
        WITH waterlooReason, waterlooChunks
        MERGE (waterlooReason)-[r:HAS_Chunk_INFO]->(waterlooChunks)
        RETURN count(r) as connected
        """,
        
        # Waterloo Combatant
        """
        MATCH (waterlooCombatant:Combatant {chunk_info: "Combatant"}), (waterlooChunks:Waterloo_Chunk)
        WHERE waterlooCombatant.chunk_info = waterlooChunks.formItem
        WITH waterlooCombatant, waterlooChunks
        MERGE (waterlooCombatant)-[r:HAS_Chunk_INFO]->(waterlooChunks)
        RETURN count(r) as connected
        """,
        
        # Waterloo Consequence
        """
        MATCH (waterlooConsequence:Consequence {chunk_info: "Consequence"}), (waterlooChunks:Waterloo_Chunk)
        WHERE waterlooConsequence.chunk_info = waterlooChunks.formItem
        WITH waterlooConsequence, waterlooChunks
        MERGE (waterlooConsequence)-[r:HAS_Chunk_INFO]->(waterlooChunks)
        RETURN count(r) as connected
        """
    ]
    
    total_connections = 0
    for query in queries:
        result = kg.query(query)
        if result:
            connections = result[0].get('connected', 0)
            total_connections += connections
    
    print(f"✓ Created {total_connections} connections between chunks and nodes!")


def create_vector_index():
    """Create vector index for efficient similarity search"""
    print("\n4. Creating vector index...")
    
    cypher = """
    CREATE VECTOR INDEX NapoleonOpenAI IF NOT EXISTS
    FOR (n:Napoleon_Chunk)
    ON n.textEmbeddingOpenAI
    OPTIONS {indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }}
    """
    
    try:
        kg.query(cypher)
        print("✓ Vector index created!")
    except Exception as e:
        print(f"⚠ Vector index creation: {e}")


def main():
    """Main import process"""
    print("\n" + "="*60)
    print("Neo4j Knowledge Graph Import Script")
    print("="*60)
    
    try:
        # Step 1: Create base structure
        create_nodes_and_relationships()
        
        # Step 2: Import chunks with embeddings
        import_chunks()
        
        # Step 3: Connect chunks to nodes
        connect_chunks_to_nodes()
        
        # Step 4: Create vector index
        create_vector_index()
        
        print("\n" + "="*60)
        print("✓ Import completed successfully!")
        print("="*60)
        print("\nYou can now run: python -m poetry run python main.py")
        
    except Exception as e:
        print(f"\n✗ Error during import: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

from neo4j import GraphDatabase
import pandas as pd

def create_knowledge_graph(uri, user, password, file_path):
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  

        df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

        for index, row in df.iterrows():
            head_name = row['head']
            tail_name = row['tail']
            relation_name = row['relation'].replace('`', '')  

            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                f"MERGE (h)-[:`{relation_name}`]->(t)"
            )
            session.run(query, head_name=head_name, tail_name=tail_name)
    
    driver.close()

# Construct EMCKG Sandbox
if __name__ == "__main__":
    uri = "your_neo4j_uri"
    username = "your_neo4j_username"
    password = "your_neo4j_password"

    file_path = './data/GenMedGPT-5K/knowledge_graph.txt'

    create_knowledge_graph(uri, username, password, file_path)




import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Neo4j knowledge graph from TSV file")
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--username', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--file', required=True, help='Path to knowledge graph TSV file')
    args = parser.parse_args()

    create_knowledge_graph(args.uri, args.username, args.password, args.file)

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
import argparse, os
from collections import defaultdict


def text2cypher(input_query: str):
    neo4j_URI = os.getenv("NEO4J_URI")
    neo4j_PASS = os.getenv("NEO4J_PASS")
    neo4j_USER = os.getenv("NEO4J_USER")
    driver = GraphDatabase.driver(neo4j_URI, auth=(neo4j_USER, neo4j_PASS))
    llm  = OpenAILLM("gpt-5")

    with driver.session() as session:
        node_property = session.run("""
        CALL db.schema.nodeTypeProperties()
        YIELD nodeLabels, propertyName, propertyTypes
        RETURN nodeLabels, propertyName, propertyTypes
        ORDER BY nodeLabels, propertyName;
    """)
        node_props = defaultdict(list)
        for val in node_property:
            node_label = val["nodeLabels"][0] if val["nodeLabels"] else "unknown"
            node_prop = val["propertyName"]
            node_prop_type = (", ").join(val["propertyTypes"])
            node_props[node_label].append(f"{node_prop}: {node_prop_type}")
        
    
        relationship_property = session.run("""
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        RETURN relType, propertyName, propertyTypes
        ORDER BY relType, propertyName;
    """)
        rel_props = defaultdict(list)
        for rec in relationship_property:
            rel = rec["relType"]
            relType = rel.strip(":`")
            prop = rec["propertyName"]
            ptype = (", ").join(rec["propertyTypes"])
            rel_props[relType].append(f"{prop}: {ptype}")

    
        rel_pattern_results = session.run("""
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS startLabels,
        type(r) AS relType,
        labels(b) AS endLabels
        ORDER BY relType, startLabels, endLabels;                          
    """)
        rel_pattern = []
        for pattern in rel_pattern_results:
            start = pattern["startLabels"][1] if pattern["startLabels"] else "?"
            rel = pattern["relType"]
            end = pattern["endLabels"][1] if pattern["endLabels"] else "?"
            rel_pattern.append(f"(:{start})-[:{rel}]->(:{end})")
    lines = []
    lines.append("Node Properties: ")

    for label, props in node_props.items():
        joined = (", ").join(props)
        lines.append(f"{label} {{{joined}}}")

    lines.append("\nRelationship Propertes")

    for rel, props in rel_props.items():
        joined = (", ").join(props)
        lines.append(f"{rel} {{{joined}}}")

    lines.append("\nThe Relationships")
    for val in rel_pattern:
        lines.append(val)

    neo4j_schema = '"""\n' + "\n".join(lines) + '\n"""'

    # (Optional) Provide user input/query pairs for the LLM to use as examples
    examples = [
    "USER INPUT: 'What reduces LDL cholestrol?' QUERY: MATCH (p:DRUG)-[:LOWERS]->(m:BIOMARKER) WHERE m.name = 'Atarovastatin' RETURN p.name"
    ]

    # Initialize the retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,  # type: ignore
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    output = retriever.search(input_query)
    print(output)
    cypher = output.metadata.get("cypher")
    answer = output.items[0]
    print(f"Cypher query of {input_query} is {output.metadata['cypher']}")
    print(f"Output answer of input query {input_query} is {output.items[0]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_query")
    args = ap.parse_args()
    text2cypher(args.input_query)

if __name__ == "__main__":
    main()


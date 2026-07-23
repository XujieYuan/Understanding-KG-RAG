import argparse
import pickle


def load_entities(file_path):
    entities = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Warning: Skipping invalid line {line_number}: {line.strip()}")
                    continue
                entities.add(parts[0])
                entities.add(parts[2])
        return list(entities)
    except FileNotFoundError:
        print(f"Error: Entity file not found: {file_path}")
        return []


def encode_and_save(items, model, output_file, item_type):
    try:
        embeddings = model.encode(items, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        data = {
            f"{item_type}": items,
            "embeddings": embeddings,
        }
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Encoded and saved {len(items)} {item_type} to {output_file}")
    except Exception as e:
        print(f"Error: Failed to encode and save {item_type}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Encode KG entities with SentenceTransformer")
    parser.add_argument('--kg', required=True, help='Path to knowledge_graph.txt (TSV: head\\trel\\ttail)')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--model', default='distiluse-base-multilingual-cased-v1',
                        help='SentenceTransformer model name or local path')
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    entities = load_entities(args.kg)
    encode_and_save(entities, model, args.output, "entities")


if __name__ == "__main__":
    main()

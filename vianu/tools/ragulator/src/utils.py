import glob
import json

import pandas as pd

import generator, retriever


def load_chunks(path_to_data, doc_names, uuid="chunk_id"):
    chunks_corpus = {}
    for doc in doc_names:
        with open(f"{path_to_data}/{doc}/{doc}.json", "r") as f:
            doc_chunks = json.load(f)
        for chunk in doc_chunks:
            chunks_corpus[str(chunk[uuid])] = chunk
    return chunks_corpus


def load_human_annotations(annotations_path, type="excel"):
    if type == "excel":
        df = pd.read_excel(annotations_path)
    elif type == "csv":
        df = pd.read_csv(annotations_path)
    else:
        raise ValueError("Type should be either 'excel' or 'csv'")
    not_chunk_columns = [col for col in df.columns if not col.startswith("Chunk")]
    df[not_chunk_columns] = df[not_chunk_columns].ffill()
    df["Chunk ID"] = df["Chunk ID"].astype(str)
    df["Chunk location - page"] = df["Chunk location - page"].astype(str)
    df["Chunk location"] = df.apply(
        lambda x: f"{x['Chunk location - document']}_page_{x['Chunk location - page']}", axis=1
    )
    return df


def load_machine_annotations(annotations_path):
    # use glob package to read all json files in the folder and then concatenate them inside a pandas dataframe
    annotations_files = glob.glob(f"{annotations_path}/*.json")
    # read all json files and concatenate them
    annotations = [json.load(open(file)) for file in annotations_files]
    col_conf = {
        "question_id": "Question ID",
        "question": "Question",
        "answer": "Answer - Ground truth",
        "content": "Chunk content",
        "chunk_id": "Chunk ID",
        "chunk_location": "Chunk location",
        "chunk_relevancy_rank": "Chunk relevancy rank",
        "doc_name": "Document name",
        "page_number": "Page number",
    }
    df = pd.DataFrame(annotations)
    # sort by document name and page number
    df = df.sort_values(by=["doc_name", "page_number"], ascending=True)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["chunk_location"] = df.apply(lambda x: f"{x['doc_name']}_page_{str(x['page_number'])}", axis=1)
    df = df[[col for col in col_conf.keys()]]
    df = df.rename(columns=col_conf)
    return df


def get_retriever(chunks_corpus, index_path, type):
    current_retriever = retriever.Retriever(corpus=chunks_corpus, index_path=index_path, type=type)
    return current_retriever


def get_generator(prompt, config, type, exclude_keys):
    if type == "remote":
        current_generator = generator.RemoteGenerator(prompt, config, type)
    elif type == "local":
        current_generator = generator.LocalGenerator(prompt, config, exclude_keys)
    else:
        raise ValueError("Invalid type. Must be either 'remote' or 'local'.")
    return current_generator

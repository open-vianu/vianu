import os
import settings
import utils


if __name__ == "__main__":
    df_annotations = utils.load_human_annotations(annotations_path=os.path.join("..", settings.PATH_TO_HUMAN_ANNOTATIONS))
    chunk_corpus = utils.load_chunks(path_to_data=os.path.join("..", settings.PATH_TO_PDF_CHUNKS), doc_names=list(settings.DOC_MAPPING.keys()))
    print(df_annotations)

from pathlib import Path

ROOT_PATH = Path(__file__).parents[2]
STATIC_PATH = ROOT_PATH / 'tools/lasa'
OUTPUT_PATH = ROOT_PATH / 'output'

DUMMY_FILEPATH = STATIC_PATH / 'dataset_dummy.csv'
FDA_FILEPATH = STATIC_PATH / 'drugsatfda_20210527.csv'
USAN_FILEPATH = STATIC_PATH / '20211125_USAN_stem_list_cumulative_csv_2.csv'
RXNORM_FILEPATH = STATIC_PATH / 'drugs_rxnorm_20210510.csv'
SWISSMEDIC_FILEPATH = STATIC_PATH / 'swissmedic_zugelassene_arzneimittel_20230531_public.xlsx'
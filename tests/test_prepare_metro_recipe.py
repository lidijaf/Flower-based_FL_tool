from data_preprocessing.recipes.metro import prepare_metro_ae_data

prepare_metro_ae_data(
    input_csv="data/raw/metro.csv",
    output_dir="data/processed/metro_ae_clients",
    num_clients=2,
)

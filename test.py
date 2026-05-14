import os
from modules.datasets import load_dataset


def inspect_columns():
    # Espande il percorso della directory dei parquet
    parquet_dir = os.path.expanduser("~/Research/datasets/librispeech-aligned_prepared")
    print(f"Ispezionando il dataset in: {parquet_dir}")

    if not os.path.exists(parquet_dir):
        print(f"ERRORE: La directory {parquet_dir} non esiste.")
        return

    try:
        # Carica il dataset (senza audio per velocità se presente)
        dataset = load_dataset("parquet", data_dir=parquet_dir)

        print("\n" + "=" * 50)
        print("STRUTTURA DEL DATASET")
        print("=" * 50)

        for split_name, split_data in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"Numero di righe: {len(split_data)}")
            print(f"Colonne: {split_data.column_names}")

            # Mostra un esempio della prima riga (escludendo campi grandi)
            print("\nEsempio (chiavi e valori della prima riga):")
            sample = split_data[0]
            for key, value in sample.items():
                if isinstance(value, (list, dict)):
                    print(f"  - {key}: {type(value)} (lunghezza: {len(value)})")
                elif isinstance(value, str) and len(value) > 100:
                    print(
                        f"  - {key}: {type(value)} (stringa lunga: {len(value)} caratteri)"
                    )
                else:
                    print(f"  - {key}: {value}")

    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")


if __name__ == "__main__":
    inspect_columns()

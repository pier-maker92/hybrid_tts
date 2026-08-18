import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

def split_parquet(input_file, output_dir, shard_size_mb, batch_size):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    max_size_bytes = shard_size_mb * 1024 * 1024

    print(f"Reading {input_path}...")
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow

    shard_idx = 0

    def get_out_path(idx):
        # Name shards as data-00000.parquet, data-00001.parquet, etc.
        return out_dir / f"data-{idx:05d}.parquet"

    current_out_path = get_out_path(shard_idx)
    writer = None

    print(f"Splitting into shards of ~{shard_size_mb}MB in {out_dir}")

    total_row_groups = parquet_file.num_row_groups
    for i in tqdm(range(total_row_groups), desc="Processing row groups"):
        # Read a row group
        table = parquet_file.read_row_group(i)
        
        # Split row group into smaller batches to check file size more frequently
        batches = table.to_batches(max_chunksize=batch_size)
        
        for batch in batches:
            batch_table = pa.Table.from_batches([batch])
            
            if writer is None:
                writer = pq.ParquetWriter(current_out_path, schema)
            
            writer.write_table(batch_table)
            
            # Check file size on disk
            if current_out_path.exists() and current_out_path.stat().st_size >= max_size_bytes:
                writer.close()
                writer = None
                shard_idx += 1
                current_out_path = get_out_path(shard_idx)

    if writer is not None:
        writer.close()

    print(f"Done! Created {shard_idx + 1} shards in {out_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Split a large parquet file into smaller shards.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the large parquet file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the shards (defaults to input file's directory)")
    parser.add_argument("--shard_size_mb", type=int, default=512, help="Max size in MB for each shard")
    parser.add_argument("--batch_size", type=int, default=250, help="Number of rows per batch for size checking")
    
    args = parser.parse_args()
    split_parquet(args.input_file, args.output_dir, args.shard_size_mb, args.batch_size)

if __name__ == "__main__":
    main()

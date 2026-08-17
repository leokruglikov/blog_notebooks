from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests


BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
OUTPUT_DIR = Path("backblaze_drive_stats")

START_YEAR = 2020
END_YEAR = 2021


def generate_urls():
    for year in range(START_YEAR, END_YEAR + 1):
        if year <= 2015:
            # Older data is stored in annual archives.
            yield f"{BASE_URL}/data_{year}.zip"
        else:
            # Newer data is stored in quarterly archives.
            for quarter in range(1, 5):
                yield f"{BASE_URL}/data_Q{quarter}_{year}.zip"


def csv_to_parquet(zip_file, csv_name, output_path):
    # Read only the header so we can assign stable column types.
    with zip_file.open(csv_name) as csv_stream:
        header = csv_stream.readline().decode("utf-8-sig").strip()
        columns = header.split(",")

    column_types = {}

    for column in columns:
        if column.startswith("smart_"):
            # SMART values may contain integers, decimals, and blanks.
            column_types[column] = pa.float64()

    # Explicit types for the main Backblaze columns.
    known_types = {
        "date": pa.string(),
        "serial_number": pa.string(),
        "model": pa.string(),
        "capacity_bytes": pa.int64(),
        "failure": pa.int64(),
    }

    for column, data_type in known_types.items():
        if column in columns:
            column_types[column] = data_type

    with zip_file.open(csv_name) as csv_stream:
        reader = pacsv.open_csv(
            csv_stream,
            read_options=pacsv.ReadOptions(
                block_size=32 * 1024 * 1024,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types=column_types,
                strings_can_be_null=True,
                null_values=["", "NA", "null"],
            ),
        )

        writer = None

        try:
            for batch in reader:
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        batch.schema,
                        compression="zstd",
                    )

                writer.write_batch(batch)

        finally:
            if writer is not None:
                writer.close()


def process_archive(url):
    archive_name = Path(url).stem
    archive_output = OUTPUT_DIR / archive_name
    archive_output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {archive_name}")

    with requests.get(url, stream=True, timeout=300) as response:
        if response.status_code == 404:
            print("  Archive does not exist, skipping")
            return

        response.raise_for_status()

        # ZIP files need random access, so temporarily save only the ZIP.
        with NamedTemporaryFile(suffix=".zip") as temp_file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    temp_file.write(chunk)

            temp_file.flush()

            with ZipFile(temp_file.name) as zip_file:
                csv_names = [
                    name
                    for name in zip_file.namelist()
                    if name.lower().endswith(".csv")
                    and not name.startswith("__MACOSX/")
                ]

                for csv_name in csv_names:
                    output_path = archive_output / (Path(csv_name).stem + ".parquet")

                    if output_path.exists():
                        print(f"  Skipping {output_path.name}")
                        continue

                    print(f"  Converting {csv_name}")
                    csv_to_parquet(
                        zip_file,
                        csv_name,
                        output_path,
                    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in generate_urls():
        try:
            process_archive(url)
        except requests.RequestException as error:
            print(f"  Download failed: {error}")
        except Exception as error:
            print(f"  Processing failed: {error}")


if __name__ == "__main__":
    main()

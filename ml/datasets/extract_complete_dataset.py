from pathlib import Path
import io
import zipfile


OUTER_ZIP = Path("data/raw/sen2venus/ALSACE.zip")
OUTPUT_DIR = Path("data/raw/sen2venus/ALSACE")


def main():
    if not OUTER_ZIP.exists():
        raise FileNotFoundError(f"Archive not found: {OUTER_ZIP}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEN2VENUS COMPLETE DATASET EXTRACTION")
    print("=" * 70)

    with zipfile.ZipFile(OUTER_ZIP, "r") as outer_zip:

        archive_names = outer_zip.namelist()

        # Find all b2b3b4b8 archives.
        inner_archives = sorted(
            name
            for name in archive_names
            if (
                "b2b3b4b8_10m.zip" in name
                or "b2b3b4b8_05m.zip" in name
            )
        )

        print(f"\nFound {len(inner_archives)} resolution archives.")

        extracted_count = 0
        skipped_count = 0

        for archive_name in inner_archives:

            filename = Path(archive_name).name

            # Example:
            # ALSACE_2018-02-14_32ULU_b2b3b4b8_10m.zip

            parts = filename.split("_")

            if len(parts) < 5:
                print(f"Skipping unexpected filename: {filename}")
                continue

            date = parts[1]

            if filename.endswith("_10m.zip"):
                resolution = "10m"
            elif filename.endswith("_05m.zip"):
                resolution = "05m"
            else:
                continue

            destination = OUTPUT_DIR / date

            # Check whether this resolution already has TIFF files.
            existing_dir = destination / "b2b3b4b8" / resolution

            if existing_dir.exists():
                tif_files = list(existing_dir.glob("*.tif"))

                if len(tif_files) > 0:
                    print(
                        f"[SKIP] {date} {resolution} "
                        f"({len(tif_files)} TIFF files already exist)"
                    )
                    skipped_count += 1
                    continue

            print(f"[EXTRACT] {filename}")

            # Read the inner ZIP from the outer ZIP.
            inner_zip_data = outer_zip.read(archive_name)

            # Open the inner ZIP directly from memory.
            with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:

                inner_zip.extractall(destination)

            extracted_count += 1

            print(f"          -> Extracted to {destination}")

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"New archives extracted: {extracted_count}")
    print(f"Archives skipped: {skipped_count}")


if __name__ == "__main__":
    main()
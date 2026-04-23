import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os

QUALITY_GRADES = ("A", "B", "C", "D", "E")


def fetch_bird_dataset(
    species_list: list[str],
    api_key: str,
    csv_output: str = "bird_metadata.csv",
    max_entries_per_species: int | None = None,
    max_quality: str | None = None,
) -> pd.DataFrame:
    
    if max_quality is not None and max_quality.upper() not in QUALITY_GRADES:
        raise ValueError(f"max_quality must be one of {QUALITY_GRADES}, got '{max_quality}'")

    base_url = "https://xeno-canto.org/api/3/recordings"
    species_to_id = {name.lower(): i for i, name in enumerate(species_list)}

    all_entries: list[dict] = []
    processed_ids: set[str] = set()

    print(f"Starting search for {len(species_list)} species...")

    for species in species_list:
        print(f"  Processing: {species}...", end=" ", flush=True)

        species_count = 0
        page = 1

        while True:
            params = {
                "query": f'sp:"{species}"',
                "key": api_key,
                "page": page,
            }

            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                break

            recordings = data.get("recordings", [])
            total_pages = int(data.get("numPages", 1))

            for rec in recordings:
                if max_entries_per_species is not None and species_count >= max_entries_per_species:
                    break

                rec_id = rec.get("id")
                if rec_id in processed_ids:
                    continue

                if max_quality is not None and QUALITY_GRADES.index(rec.get("q", "E").upper()) > QUALITY_GRADES.index(max_quality.upper()):
                    continue

                labels: list[int] = []

                main_species = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip().lower()
                if main_species in species_to_id:
                    labels.append(species_to_id[main_species])

                for bg in rec.get("also", []):
                    bg_lower = bg.lower()
                    if bg_lower in species_to_id:
                        lbl = species_to_id[bg_lower]
                        if lbl not in labels:
                            labels.append(lbl)

                if not labels:
                    continue

                file_url = rec.get("file", "")
                if file_url.startswith("//"):
                    file_url = "https:" + file_url

                all_entries.append({
                    "url": file_url,
                    "label_id": ",".join(map(str, sorted(labels)))
                })
                processed_ids.add(rec_id)
                species_count += 1

            cap_reached = max_entries_per_species is not None and species_count >= max_entries_per_species
            if cap_reached or page >= total_pages:
                break

            page += 1
            time.sleep(1.1)

        print(f"done ({species_count} entries kept).")
        time.sleep(1.1)

    df = pd.DataFrame(all_entries)
    df.to_csv(csv_output, index=False)

    print(f"\nDone. CSV saved to: {csv_output}")
    print(f"Total entries: {len(df)}")
    return df


# --- CONFIG ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
SPECIES = [
    "Larus fuscus",
    "Parus major",
    "Troglodytes troglodytes",
    "Fringilla coelebs",
    "Turdus merula",
    "Erithacus rubecula",
    "Phylloscopus collybita",
    "Columba palumbus",
    "Picus viridis",
    "Alauda arvensis"
]

fetch_bird_dataset(
    species_list=SPECIES,
    api_key=API_KEY,
    max_entries_per_species=250,
    max_quality="B",
)
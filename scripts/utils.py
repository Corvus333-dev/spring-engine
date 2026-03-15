import json
from pathlib import Path

# Create project root path relative to this module
ROOT = Path(__file__).resolve().parent.parent

def _search_species(query):
    """Returns matching species metadata entries"""
    species_file = ROOT / 'data' / 'phenology' / 'metadata' / 'species.json'

    try:
        with open(species_file, 'r', encoding='utf-8') as f:
            species_metadata = json.load(f)
    except FileNotFoundError as e:
        e.add_note("Species metadata missing. Run 'download_phenology_metadata()'")
        raise

    query = query.lower()
    matches = []

    for s in species_metadata:
        scope = f"{s['common_name']} {s['genus']} {s['species']}".lower()
        if query in scope:
            matches.append({
                'species_id': s['species_id'],
                'common_name': s['common_name'],
                'scientific_name': f"{s['genus']} {s['species']}"
            })

    return matches

def lookup_species(query=None):
    """
    Searches species metadata for entries whose common or scientific name contains the query and prints matching IDs.

    Args:
        query (str | None): Species name to search for. Defaults to None.

    Raises:
        FileNotFoundError: If species metadata file is missing. Run `pipeline.download_phenology_metadata()` if needed.
    """
    if query is None:
        query = input("Enter species name: ")

    matches = _search_species(query)

    if not matches:
        print("No matches found")
        return

    col_0_header = "ID"
    col_1_header = "Common Name (Scientific Name)"
    col_0_width = max(len(col_0_header), max(len(str(s['species_id'])) for s in matches))
    col_1_width = max(len(col_1_header), max(len(f"{s['common_name']} ({s['scientific_name']})") for s in matches))
    row_break = "=" * (col_0_width + col_1_width + 3)

    print(row_break)
    print(f"{col_0_header:>{col_0_width}} | {col_1_header}")
    print(row_break)

    for s in matches:
        print(f"{s['species_id']:>{col_0_width}} | {s['common_name']} ({s['scientific_name']})")
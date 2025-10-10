import json, requests
from tqdm import tqdm

ensembl_ids = list(json.load(open("../data_utils/vocab/gene_ranking_map.json")))
url = "https://api.platform.opentargets.org/api/v4/graphql"
query = """query targetInfo($ids: [String!]!) {
  targets(ensemblIds: $ids) { id biotype }
}"""

biotypes = {}
batch_size = 1000
for i in tqdm(range(0, len(ensembl_ids), batch_size), desc="Fetching biotypes"):
    batch = ensembl_ids[i:i+batch_size]
    try:
        r = requests.post(url, json={"query": query, "variables": {"ids": batch}}, timeout=60)
        if r.ok:
            data = r.json().get("data", {}).get("targets", [])
            for t in data:
                biotypes[t["id"]] = t.get("biotype")
        for eid in batch:
            biotypes.setdefault(eid, None)
    except KeyboardInterrupt:
        raise
    except Exception:
        for eid in batch:
            biotypes[eid] = None

json.dump(biotypes, open("../data_utils/vocab/gene_biotypes.json", "w"))

# %%

import logging
logging.basicConfig(level=logging.INFO)

# %%

from SPARQLWrapper import SPARQLWrapper, JSON, POST

from typing import Optional

import pandas as pd

# %%

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
# Use POST for queries to avoid long GET URLs / large headers (HTTP 431)
# Ensure we use POST for this request (helps with very large queries)
sparql.setMethod(POST)

# %%

def execute_query(sparql: SPARQLWrapper, query: str) -> Optional[list]:
    sparql.setQuery(query)
    records = []
    try:
        response = sparql.queryAndConvert()
        for result in response["results"]["bindings"]:
            record = {}
            for key in result:
                if result[key]["type"] == "uri":
                    record[key] = result[key]["value"].split("/")[-1]
                else:
                    record[key] = result[key]["value"]
            records.append(record)
        return records
    except Exception as e:
        logging.warning(e)
        return None

def get_entities(entity_id: str, limit=1, offset=0):
    query = f"""
    SELECT DISTINCT ?entity
    WHERE {{
        ?entity wdt:P31 wd:{entity_id} .
    }}
    LIMIT {limit} OFFSET {offset}
    """
    results = execute_query(sparql, query)
    return results

def get_human_per_citizenship(country_ids: list[str], limit=1, offset=0):
    # 1. Build a UNION block for each country
    unions = []
    for country_id in country_ids:
        subquery = f"""
        {{
            # Added ?citizenship here so the outer query can see it
            SELECT ?entity ?citizenship
            WHERE {{
                BIND(wd:{country_id} AS ?citizenship) .
                ?entity wdt:P31 wd:Q5 .
                ?entity wdt:P27 ?citizenship .
                
                # Only select people who have at least 20 Wikipedia articles
                ?entity wikibase:sitelinks ?sitelinks .
                FILTER(?sitelinks > 20) .
            }}
            LIMIT {limit} OFFSET {offset}
        }}
        """
        unions.append(subquery)
    
    # 2. Join them all together into one single network request
    combined_query = f"""
    SELECT ?entity ?citizenship
    WHERE {{
        {" UNION ".join(unions)}
    }}
    """
    # 3. Execute the combined query
    results = execute_query(sparql, combined_query)
    return results

def get_entity_aliases(entity_ids: list):
    entity_ids_str = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
    query = f"""
    SELECT ?entity ?entityLabel ?alias
    WHERE {{
        VALUES ?entity {{ {entity_ids_str} }}
        ?entity skos:altLabel ?alias .
        FILTER(LANG(?alias) = "en")
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    labels = execute_query(sparql, query)
    return labels

def merge_attribute(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, attribute_column: str) -> pd.DataFrame:
    merged = left.merge(right, left_on=left_on, right_on=right_on, how="left")
    merged = merged.drop(columns=[right_on])
    merged = merged.rename(columns={attribute_column: f"{left_on}_{attribute_column}"})
    return merged

# %%

if __name__ == "__main__":
    # query = """
    # SELECT ?child
    # WHERE {
    # #   ?child  father   Bach
    #     ?child  wdt:P22  wd:Q1339 .
    # }
    # """
    country_ids = ["Q183", "Q142", "Q27"]
    country_aliases = get_entity_aliases(country_ids)
    entities = get_human_per_citizenship(country_ids, limit=200, offset=0)
    human_ids = set([entity["entity"] for entity in entities])
    human_aliases = get_entity_aliases(human_ids)
    
# %%

if __name__ == "__main__":
    df = pd.DataFrame(entities)
    df["r1"] = "P27"
    df = df.rename(columns={"entity": "e1", "citizenship": "e2"})
    country_aliases_df = pd.DataFrame(country_aliases).groupby("entity")["alias"].apply(list).reset_index(name="aliases")
    df = merge_attribute(df, country_aliases_df, left_on="e2", right_on="entity", attribute_column="aliases")
    country_labels_df = pd.DataFrame(country_aliases).groupby("entity")["entityLabel"].first().reset_index(name="label")
    df = merge_attribute(df, country_labels_df, left_on="e2", right_on="entity", attribute_column="label")
    human_aliases_df = pd.DataFrame(human_aliases).groupby("entity")["alias"].apply(list).reset_index(name="aliases")
    df = merge_attribute(df, human_aliases_df, left_on="e1", right_on="entity", attribute_column="aliases")
    human_labels_df = pd.DataFrame(human_aliases).groupby("entity")["entityLabel"].first().reset_index(name="label")
    df = merge_attribute(df, human_labels_df, left_on="e1", right_on="entity", attribute_column="label")
    df = df[["e1", "e1_label", "e1_aliases", "r1", "e2", "e2_label", "e2_aliases"]]
    df = df.dropna()
    df.to_csv("entity_citizenship.csv", index=False)

# %%

if __name__ == "__main__":
    import ast
    df = pd.read_csv("entity_citizenship.csv")
    df["e1_aliases"] = df["e1_aliases"].apply(ast.literal_eval)
    df["e2_aliases"] = df["e2_aliases"].apply(ast.literal_eval)
    df["e2_answers"] = df.apply(lambda row: [row["e2_label"]] + row["e2_aliases"], axis=1)

# %%

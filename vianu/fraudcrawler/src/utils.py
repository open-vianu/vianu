



def transform_df_to_json(query_info, df):
    # Transform the DataFrame into a list of dictionaries
    recovered_urls = []

    for _, row in df.iterrows():
        recovered_urls.append({
            "offerRoot": "EBAY",  # Assuming "EBAY" is constant
            "status": "Initial",
            "url": row["url"],
            "price": row.get("product.price", "N/A"),
            "title": row.get("product.name", ""),
            "fullDescription": row.get("product.description", "N/A"),
            "images": row.get("product.images", [])
        })

    # Combine query and results into a final JSON structure
    output_json = {
        "query": query_info,
        "recovered_urls": recovered_urls
    }
    return output_json


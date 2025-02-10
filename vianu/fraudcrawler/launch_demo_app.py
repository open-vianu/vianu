import gradio as gr
from vianu.fraudcrawler.src.client import FraudCrawlerClient

# Sample JSON data for three example drugs with valid image URLs
iron_magic_milk_data = [
    {
        "offerRoot": "GOOGLE",
        "url": "https://www.sport-enzinger.com/kategorie/ernaehrung/aminosaeuren/komplexe-aminos/",
        "title": "/ komplexe Aminos",
        "price": "12.50 \u20ac",
        "zyteExecutionTime": 3.734952926635742,
        "zyteProbability": 0.005122258793562651,
        "fullDescription": "Amazing product sold here",
        "images": [
            "https://www.sport-enzinger.com/wp-content/uploads/2022/08/12914_95bb945f1ece-600x600.webp.jpg",
        ],
        "pageType": "other",
    },
    {
        "offerRoot": "GOOGLE",
        "url": "https://www.labelhair.at/milkshake/sos-roots-mahogany",
        "price": "16.59 \u20ac",
        "title": "SOS Roots MAHOGANY, 75 ml",
        "fullDescription": "SOS Roots MAHAGONY von milk_shake ist ein nat\u00fcrliches Spray, das den Ansatz bis zu der n\u00e4chsten Haarw\u00e4sche in sattem Mahagoni kaschiert. Der feine, schnell trocknende Spr\u00fchnebel erm\u00f6glicht, dass das Haar sauber und trocken bleibt, ohne es zu beschweren. Die nat\u00fcrlichen Pigmente basieren auf Mineralien und sogen f\u00fcr eine intensivere Farbe und eine optimale Grauabdeckung.",
        "zyteExecutionTime": 2.3590247631073,
        "zyteProbability": 0.9918913841247559,
        "images": [
            "https://la.nice-cdn.com/upload/image/product/large/default/13228_d1afc42f.1024x1024.jpg"
        ],
        "pageType": "ecommerce_product",
    },
    {
        "offerRoot": "GOOGLE",
        "url": "https://www.basler-beauty.at/marken/milk-shake/milk-shake-light-catcher-fast-toning.html",
        "price": "5.9 \u20ac",
        "title": "milk_shake Light Catcher fast toning T\u00f6nung",
        "fullDescription": "5 NEUE FARBT\u00d6NE F\u00dcR TRAUMHAFT BLONDES HAAR. Alle Farbt\u00f6ne sind untereinander mischbar!\n\nEntdecken Sie die neuen milk_shake \u00ae light catcher fast toning Direktfarben, die das Light-Catcher-Sortiment bereichern und vervollst\u00e4ndigen. Empfohlen f\u00fcr Stufe 9 oder helleres blondes und augehelltes Haar.\n\n5 Farbt\u00f6ne f\u00fcr bereits aufgehelltes Haar:\n\nSofortige und zarte Farbt\u00f6ne\nDirektpigmente, die 4\u20135 Haarw\u00e4schen* halten\npflegt und stellt den nat\u00fcrlichen pH-Wert des Haares wieder her\nMit hochwertigen Pigmenten frei von Ammoniak und Oxidationsmitteln, Parabenen (SLS und SLES)\nAuffrischung der Haarfarbe zwischen den T\u00f6nungsbehandlungen\nErzeugen zarter Pastelleffekte, Verst\u00e4rken und Korrigieren von Farbt\u00f6nen\n\n*je nach Haarzustand, Einwirkzeit und Menge des verwendeten Produkts\n\nAnwendung: Gleichm\u00e4\u00dfig im sauberen, feuchten Haar verteilen oder je nach gew\u00fcnschtem Ergebnis auf ausgew\u00e4hlte Str\u00e4hnen auftragen. Nach der Einwirkzeit aufsch\u00e4umen und aussp\u00fclen. Falls erforderlich, mit einem Conditioner nachbehandeln. Einwirkzeit: zwischen 5 und 20 Minuten. Die Farbt\u00f6ne k\u00f6nnen einem Shampoo, einer Maske oder einer Pflegesp\u00fclung beigef\u00fcgt werden, um einen sanften Toning-Effekt zu erzielen.",
        "zyteExecutionTime": 3.5089142322540283,
        "zyteProbability": 0.7695555090904236,
        "images": [
            "https://cdn.basler-beauty.de/out/pictures/generated/product/1/980_980_100/40285fef8d320087018d63f40edc5f83-milk-shake-Light-Catcher-fast-toning.68603879.jpg",
            "https://cdn.basler-beauty.de/out/pictures/generated/manufacturer/icon/100_100_100/milk-shake.b1107237.png",
            "https://cdn.basler-beauty.de/out/pictures/generated/product/1/1200_1200_100/40285fef8d320087018d63f40edc5f83-milk-shake-Light-Catcher-fast-toning.68603879.jpg",
        ],
        "pageType": "ecommerce_product",
    },
    {
        "offerRoot": "GOOGLE",
        "url": "https://marienapotheke.at/en/produkt/oleovital-eisen-classic-12mg/?add-to-cart=34814",
        "price": "28.6 \u20ac",
        "title": "Oleovital Iron Classic 12 mg sachets",
        "fullDescription": "Iron contributes to normal cognitive function.\n\nIron supports the formation of red blood cells and hemoglobin.\n\nIron supports oxygen transport and folic acid, vitamin B6, vitamin B12 and vitamin C reduce symptoms of tiredness.\n\nIron supports the energy metabolism.\n\nContents: 30 sachets with cola flavor",
        "zyteExecutionTime": 3.840689182281494,
        "zyteProbability": 0.9885560870170593,
        "images": [
            "https://marienapotheke.at/wp-content/uploads/2024/08/OLEOvital_Eisen_classic_ergebnis.webp"
        ],
        "pageType": "ecommerce_product",
    },
]

devalife_data = [
    {
        "offerRoot": "GOOGLE",
        "url": "https://www.medizinfuchs.at/devalife-kapseln.html",
        "title": "Alle Angebote zu Devalife Kapseln",
        "fullDescription": "26 Angebote zu Devalife kapseln im Medikamenten Preisvergleich. Devalife kapseln g\u00fcnstig kaufen und sparen bei medizinfuchs.at",
        "zyteExecutionTime": 3.1472229957580566,
        "zyteProbability": 0.005644466727972031,
        "images": ["https://www.medizinfuchs.at/images/medizinfuchs_logo_big.png"],
        "price": "16.59 \u20ac",
        "pageType": "other",
    },
    {
        "offerRoot": "EBAY",
        "url": "https://www.ebay.at/itm/166811727812",
        "price": "60.0 EUR",
        "title": "DEVALIFE 1 Packungen 30 Kapsel 1 Monate Kur",
        "fullDescription": "Der Verk\u00e4ufer ist f\u00fcr dieses Angebot verantwortlich.\n\neBay-Artikelnr.:166811727812\n\nArtikelmerkmale\n\nArtikelzustand\nNeu: Neuer, unbenutzter und unbesch\u00e4digter Artikel in nicht ge\u00f6ffneter Originalverpackung (soweit ... Mehr erfahren\u00dcber den ZustandNeu: Neuer, unbenutzter und unbesch\u00e4digter Artikel in nicht ge\u00f6ffneter Originalverpackung (soweit eine Verpackung vorhanden ist). Die Verpackung sollte der im Einzelhandel entsprechen. Ausnahme: Der Artikel war urspr\u00fcnglich in einer Nichteinzelhandelsverpackung verpackt, z. B. unbedruckter Karton oder Plastikh\u00fclle. Weitere Einzelheiten im Angebot des Verk\u00e4ufers. Alle Zustandsdefinitionen ansehenwird in neuem Fenster oder Tab ge\u00f6ffnet\n\nHerstellernummer\nnicht zutreffend\n\nMarke\nDevalife\n\nFormulierung\nKapseln\n\nHerstellungsland und -region\nT\u00fcrkei\n\nAngebotspaket\nJa\n\nMindesthaltbarkeitsdatum\n2025\n\nWirksame Inhaltsstoffe\nHimbeere, Ballaststoff, Gr\u00fcntee-Extrakt, Guarana, Fenchel, Koriander, Heidekrautbl\u00e4tter, Rosmarin\n\nArtikelbeschreibung des Verk\u00e4ufers",
        "zyteExecutionTime": 2.830410957336426,
        "zyteProbability": 0.9628998041152954,
        "images": [
            "https://i.ebayimg.com/images/g/VbwAAOSwy1NaZQPn/s-l500.webp",
            "https://i.ebayimg.com/images/g/sHQAAOSwk-hkDaYx/s-l500.webp",
            "https://i.ebayimg.com/images/g/WlYAAOSw0txkDaYy/s-l140.webp",
            "https://i.ebayimg.com/images/g/VbwAAOSwy1NaZQPn/s-l140.webp",
            "https://i.ebayimg.com/images/g/ayAAAOSwjVhkDaYz/s-l500.webp",
            "https://i.ebayimg.com/images/g/ayAAAOSwjVhkDaYz/s-l1600.webp",
            "https://i.ebayimg.com/images/g/UcUAAOSwoRBaZQVk/s-l140.webp",
            "https://i.ebayimg.com/images/g/Rn0AAOSwIk9aZQU0/s-l140.webp",
            "https://i.ebayimg.com/images/g/WlYAAOSw0txkDaYy/s-l500.webp",
            "https://i.ebayimg.com/images/g/HD8AAOSwZFJkDaXo/s-l140.webp",
            "https://i.ebayimg.com/images/g/ayAAAOSwjVhkDaYz/s-l960.webp",
            "https://i.ebayimg.com/images/g/HD8AAOSwZFJkDaXo/s-l960.webp",
            "https://i.ebayimg.com/images/g/Rn0AAOSwIk9aZQU0/s-l960.webp",
            "https://i.ebayimg.com/images/g/ayAAAOSwjVhkDaYz/s-l140.webp",
            "https://i.ebayimg.com/images/g/WlYAAOSw0txkDaYy/s-l960.webp",
            "https://i.ebayimg.com/images/g/VbwAAOSwy1NaZQPn/s-l960.webp",
            "https://i.ebayimg.com/images/g/KJUAAOSwoFtkDaY~/s-l140.webp",
            "https://i.ebayimg.com/images/g/UcUAAOSwoRBaZQVk/s-l500.webp",
            "https://i.ebayimg.com/images/g/sHQAAOSwk-hkDaYx/s-l140.webp",
            "https://i.ebayimg.com/images/g/Zp8AAOSwEMpkDaX1/s-l500.webp",
            "https://i.ebayimg.com/images/g/KJUAAOSwoFtkDaY~/s-l500.webp",
            "https://i.ebayimg.com/images/g/Zp8AAOSwEMpkDaX1/s-l140.webp",
            "https://i.ebayimg.com/images/g/Rn0AAOSwIk9aZQU0/s-l1600.webp",
            "https://i.ebayimg.com/images/g/KJUAAOSwoFtkDaY~/s-l1600.webp",
            "https://i.ebayimg.com/images/g/sHQAAOSwk-hkDaYx/s-l1600.webp",
            "https://i.ebayimg.com/images/g/Zp8AAOSwEMpkDaX1/s-l1600.webp",
            "https://i.ebayimg.com/images/g/UcUAAOSwoRBaZQVk/s-l960.webp",
            "https://i.ebayimg.com/thumbs/images/g/KJUAAOSwoFtkDaY~/s-l500.jpg",
            "https://i.ebayimg.com/images/g/UcUAAOSwoRBaZQVk/s-l1600.webp",
            "https://i.ebayimg.com/images/g/VbwAAOSwy1NaZQPn/s-l1600.webp",
            "https://i.ebayimg.com/images/g/HD8AAOSwZFJkDaXo/s-l500.webp",
            "https://i.ebayimg.com/images/g/sHQAAOSwk-hkDaYx/s-l960.webp",
            "https://i.ebayimg.com/images/g/Rn0AAOSwIk9aZQU0/s-l500.webp",
            "https://i.ebayimg.com/images/g/WlYAAOSw0txkDaYy/s-l1600.webp",
            "https://i.ebayimg.com/images/g/Zp8AAOSwEMpkDaX1/s-l960.webp",
            "https://i.ebayimg.com/images/g/HD8AAOSwZFJkDaXo/s-l1600.webp",
        ],
        "pageType": "ecommerce_product",
    },
]


# Function to process the input and display the JSON data
def display_results(site_token, serp_token, country, search_term, selected_example):
    # Check if any of the custom configuration fields are filled
    if site_token or serp_token or search_term:
        return "<h3>Reach out to us if you want to know more about how to use this sandbox environment</h3>"

    # Select the example data based on the dropdown choice
    if selected_example == "Devalife":
        data = devalife_data
    elif selected_example == "Iron Magic Milk":
        data = iron_magic_milk_data
    else:
        return "Please select a valid example."

    # Generate HTML content
    html_content = ""
    for item in data:
        image_url = item["images"][0] if item["images"] else ""
        html_content += f"""
        <div style='border:1px solid #ccc; padding:10px; margin-bottom:10px;'>
            <h2>{item["title"]}</h2>
            <img src='{image_url}' alt='Image' style='max-width:200px;'/>
            <p><strong>Price:</strong> {item["price"]}</p>
            <p><strong>Description:</strong> {item["fullDescription"]}</p>
            <p><a href='{item["url"]}'>View Product</a></p>
        </div>
        """
    return html_content


def handle_inputs(site_token, serp_token, search_term, selected_example):
    # Validate input completeness
    if len(site_token) > 10 and len(serp_token) > 10 and len(search_term) > 3:
        # Instantiate the client
        nc_client = FraudCrawlerClient()
        nc_client.serpapi_token = serp_token
        nc_client.zyte_api_key = site_token

        try:
            # Perform the search using FraudCrawler pipeline
            df = nc_client.search(search_term, num_results=8, location="Switzerland")

            # Convert DataFrame to JSON
            search_results = df.to_dict(orient="records")

            # Generate HTML content from JSON
            html_content = ""
            if len(search_results) == 0:
                html_content = "<h3 style='color:green;'>Your search did not retrieve any results with the default configuration.</h3>"
            else:
                for item in search_results:
                    title = item.get("product.name", "No title")
                    price = item.get("product.price", "Price not available")
                    description = item.get(
                        "product.description", "No description available"
                    )
                    url = item.get("url", "#")
                    image_url = item.get("product.mainImage.url", "")

                    html_content += f"""
                    <div style='border:1px solid #ccc; padding:10px; margin-bottom:10px;'>
                        <h2>{title}</h2>
                        <img src='{image_url}' alt='Image' style='max-width:200px;'/>
                        <p><strong>Price:</strong> {price}</p>
                        <p><strong>Description:</strong> {description}</p>
                        <p><a href='{url}'>View Product</a></p>
                    </div>
                    """
            return html_content
        except Exception as e:
            return f"<h3 style='color:red;'>Error: {str(e)}, please check your credentials or reach out to us at hello@vianu.com.</h3>"
    elif len(site_token) > 0 or len(serp_token) > 0 or len(search_term) > 0:
        return "<h3 style='color:red;'>Please fill out all the required credentials (Zyte API, Serp API, and Search Term).</h3>"
    else:
        # Provide example data if tokens are missing
        if selected_example == "Devalife":
            data = devalife_data
        elif selected_example == "Iron Magic Milk":
            data = iron_magic_milk_data
        else:
            return "Please select a valid example."

        # Generate HTML content for example data
        html_content = ""
        for item in data:
            image_url = item["images"][0] if item["images"] else ""
            html_content += f"""
            <div style='border:1px solid #ccc; padding:10px; margin-bottom:10px;'>
                <h2>{item["title"]}</h2>
                <img src='{image_url}' alt='Image' style='max-width:200px;'/>
                <p><strong>Price:</strong> {item["price"]}</p>
                <p><strong>Description:</strong> {item["fullDescription"]}</p>
                <p><a href='{item["url"]}'>View Product</a></p>
            </div>
            """
        return html_content


# Define the Gradio app layout
with gr.Blocks(title="FraudCrawler Sandbox") as app:
    gr.Markdown(
        """
        <h1 style="text-align: center;">FraudCrawler Sandbox</h1>
        """,
        elem_id="centered-title",
    )
    gr.Markdown("""
    Welcome to the FraudCrawler Sandbox! This tool allows you to explore how our search pipeline works using real-life examples or by performing your own custom searches. 

    - **Examples**: Choose from preloaded data to see how the results are displayed.
    - **Custom Search**: Input your search term along with a valid SERP API token and Site token. You can obtain these tokens for free by registering on their respective websites.

    If you have any questions, feel free to reach out to us at hello@vianu.org. We're always happy to help! 
    """)
    with gr.Row():
        with gr.Column(scale=1):
            # Dropdown for examples
            gr.Markdown("### Examples")
            example_dropdown = gr.Dropdown(
                label="Select Example Drug",
                choices=["Devalife", "Iron Magic Milk"],
                value="Devalife",
                interactive=True,
            )

            # Inputs for API tokens and search terms
            gr.Markdown("### Custom Configuration")
            with gr.Accordion("Search for your own keywords", open=False):
                zyte_api = gr.Textbox(
                    label="Zyte API Token",
                    type="password",
                    placeholder="Enter Zyte API Token",
                )
                serp_api = gr.Textbox(
                    label="Serp API Token",
                    type="password",
                    placeholder="Enter Serp API Token",
                )
                search_term = gr.Textbox(
                    label="Search Term", placeholder="Enter your search term"
                )

        # Results display section
        with gr.Column(scale=2):
            results = gr.HTML(label="Results")

    # Button to trigger display logic
    display_button = gr.Button("Display Results")
    display_button.click(
        fn=handle_inputs,
        inputs=[zyte_api, serp_api, search_term, example_dropdown],
        outputs=results,
    )


def main():
    app.launch(
        debug=True
    )  # Add share=True if you want to create a 72h lasting demo deployment.


# Launch the app
if __name__ == "__main__":
    main()

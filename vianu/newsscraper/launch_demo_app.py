# main.py (Gradio app)

import os
import gradio as gr
import logging
from vianu.newsscraper.src.db_manager import DatabaseManager
from vianu.newsscraper.src.scrape import start_scraper
import plotly.graph_objects as go  # Import Plotly for charting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

if os.getenv("DEPLOYMENT_MODE") == "cloud":
    # Using cloud database:
    cloud_connection = os.getenv("SQLIGHTCONNECTIONSTRING")
    db = DatabaseManager(db_mode="cloud", cloud_connection_string=cloud_connection)
else:
    # Using the local database
    DATABASE_PATH = "./data/database.db"
    db = DatabaseManager(db_mode="local", local_db_path=DATABASE_PATH)


def list_terms():
    return db.list_terms()


def list_articles(search_query=None):
    return db.get_articles(search_query=search_query)


def get_statistics():
    terms = list_terms()
    articles = list_articles()
    summary_md = f"""
    <div style="display:flex; gap:2rem; justify-content:center; align-items:center; margin:20px 0;">
        <div style="text-align:center;">
            <div style="font-size:2em; font-weight:bold;">{len(terms)}</div>
            <div style="font-size:1em; color:#555;">Total Terms</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:2em; font-weight:bold;">{len(articles)}</div>
            <div style="font-size:1em; color:#555;">Total Articles</div>
        </div>
    </div>
    """
    return summary_md


def get_latest_metadata():
    return db.get_latest_metadata()


def refresh_metadata():
    metadata = get_latest_metadata()
    if metadata:
        return f"""
        <div style="margin-top: 20px;">
            <h3>Last Scraping Run:</h3>
            <p><strong>Timestamp:</strong> {metadata["last_scraping_timestamp"]}</p>
            <p><strong>Status:</strong> {metadata["last_scraping_status"]}</p>
        </div>
        """
    else:
        return "<div style='margin-top: 20px;'><p>No scraping run has been executed yet.</p></div>"


def generate_pill_html(terms):
    pill_html = '<div class="pill-container">'
    for term in terms:
        pill_html += f'<span class="pill">{term}</span>'
    pill_html += "</div>"
    return pill_html


def add_term_and_scrape(term):
    term = term.strip()
    if term == "":
        return (
            "Please enter a valid search term.",
            list_articles(),
            generate_pill_html(list_terms()),
            get_statistics(),
            get_insights(),
            gr.update(choices=list_terms(), value=None),  # Update Dropdown choices
        )

    db.insert_terms([term])
    pages = [1, 2]
    start_scraper(pages, [term])

    updated_terms = list_terms()
    return (
        f"Scraping done for: {term}",
        list_articles(),
        generate_pill_html(updated_terms),
        get_statistics(),
        get_insights(),
        gr.update(choices=updated_terms, value=None),  # Update Dropdown choices
    )


def delete_term(term):
    term = term.strip()
    if not term:
        return (
            "No term selected to delete.",
            list_articles(),
            generate_pill_html(list_terms()),
            get_statistics(),
            get_insights(),
            gr.update(choices=list_terms(), value=None),  # Update Dropdown choices
        )

    db.delete_term(term)
    updated_terms = list_terms()
    return (
        f"Deleted term: {term}",
        list_articles(),
        generate_pill_html(updated_terms),
        get_statistics(),
        get_insights(),
        gr.update(choices=updated_terms, value=None),  # Update Dropdown choices
    )


def refresh_data():
    articles = list_articles()
    pills = generate_pill_html(list_terms())
    stats = get_statistics()
    insights = get_insights()
    return articles, pills, stats, insights


def search_and_update(query):
    articles = list_articles(query)
    terms = list_terms()
    stats = get_statistics()
    insights = get_insights()
    return articles, generate_pill_html(terms), stats, insights


def reset_search():
    articles = list_articles()
    terms = list_terms()
    stats = get_statistics()
    insights = get_insights()
    return articles, generate_pill_html(terms), stats, insights


def get_term_counts():
    """Fetches the count of each term from the database."""
    terms = list_terms()
    articles = list_articles()
    term_counts = {term: 0 for term in terms}
    for article in articles:
        try:
            term = article[0]  # Access 'Search Term' by index
            if term in term_counts:
                term_counts[term] += 1
        except IndexError:
            logging.error(f"Malformed article data: {article}")
            continue
    return term_counts


def get_insights():
    """Generates a horizontal bar chart of term counts."""
    term_counts = get_term_counts()
    terms = list(term_counts.keys())
    counts = list(term_counts.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=terms,
                orientation="h",
                marker=dict(color="rgba(55, 128, 191, 0.7)"),
            )
        ]
    )
    fig.update_layout(
        title="",
        xaxis_title="Count",
        yaxis_title="Search Terms",
        yaxis=dict(
            autorange="reversed",  # To display the highest count on top
            tickmode="linear",  # Ensure all labels are shown
        ),
        template="plotly_white",
        height=900,
    )
    return fig


css = """
<style>
.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 10px;
}
.pill {
    background-color: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 16px;
    padding: 5px 10px;
    font-size: 14px;
    color: #333;
    cursor: default;
    display: inline-flex;
    align-items: center;
    margin: 3px;
}
.pill:hover {
    background-color: #e0e0e0;
}
.search-container {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}
.message {
    margin-top: 10px;
    color: green;
    font-weight: bold;
}
</style>
"""


def load_all_data():
    articles, pills, stats, insights = refresh_data()
    terms = list_terms()
    metadata = refresh_metadata()
    return (
        articles,
        pills,
        stats,
        gr.update(choices=terms, value=None),
        metadata,
        insights,
    )


with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Parlament Scraper") as demo:
    gr.Markdown("# Parlament-Ticker")
    gr.Markdown(
        "The Parlament-Scraper by vianu searches in the [news section](https://www.parlament.ch/de/services/suche-news), the [business database](https://www.parlament.ch/de/ratsbetrieb/suche-curia-vista), and the [official bulletin](https://www.parlament.ch/de/ratsbetrieb/suche-amtliches-bulletin) on parlament.ch for the search terms entered above."
    )
    gr.Markdown("The app triggers an automatic refresh at 3am.")
    metadata_html = gr.HTML(label="Latest Scraping Metadata")
    statistics_markdown = gr.Markdown()
    gr.Markdown("### Following Search Terms are Stored:")
    terms_html = gr.HTML(label="Saved Search Terms")

    with gr.Row():
        with gr.Column(scale=2):
            search_input = gr.Textbox(
                label="Search Articles", placeholder="search...", interactive=True
            )
        with gr.Column(scale=1):
            search_button = gr.Button("Search")
        with gr.Column(scale=1):
            reset_button = gr.Button("Reset")

    with gr.Accordion("Add or remove Terms", open=False):
        new_term = gr.Textbox(
            label="New Search Term", placeholder="Enter a search term like 'heilmittel'"
        )
        add_button = gr.Button("Add and Scrape")

        selected_term = gr.Dropdown(label="Select a Term to Delete", choices=[])
        delete_term_button = gr.Button("Delete Selected Term")

    # New Insights Accordion
    with gr.Accordion("Insights", open=False):
        insights_plot = gr.Plot(label="Term Counts Bar Chart")

    articles_table = gr.Dataframe(
        headers=["Search Term", "Date", "Name", "URL"],
        datatype=["str", "str", "str", "str"],
        wrap=True,
        value=[],
    )

    message_display = gr.Markdown(value="", visible=True, label="Message")

    # Event Handlers
    add_button.click(
        add_term_and_scrape,
        inputs=[new_term],
        outputs=[
            message_display,
            articles_table,
            terms_html,
            statistics_markdown,
            insights_plot,
            selected_term,
        ],
    )

    delete_term_button.click(
        delete_term,
        inputs=[selected_term],
        outputs=[
            message_display,
            articles_table,
            terms_html,
            statistics_markdown,
            insights_plot,
            selected_term,
        ],
    )

    search_button.click(
        search_and_update,
        inputs=[search_input],
        outputs=[articles_table, terms_html, statistics_markdown, insights_plot],
    )

    reset_button.click(
        reset_search,
        inputs=[],
        outputs=[articles_table, terms_html, statistics_markdown, insights_plot],
    )

    # Load Initial Data
    demo.load(
        load_all_data,
        inputs=[],
        outputs=[
            articles_table,
            terms_html,
            statistics_markdown,
            selected_term,
            metadata_html,
            insights_plot,
        ],
    )

if __name__ == "__main__":
    demo.launch()

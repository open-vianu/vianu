from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import gradio as gr
import pandas as pd

from vianu import BaseApp
from vianu.lasa.settings import LOG_LEVEL, PROCESSES
from vianu.lasa.settings import (
    GRADIO_APP_NAME,
    GRADIO_RESULTS_COLUMNS,
    GRADIO_RESULTS_COLUMNS_DEFAULT,
)
from vianu.lasa.settings import SOURCES, THRESHOLD
from vianu.lasa.__main__ import main

logger = logging.getLogger(__name__)

# App settings
_ASSETS_PATH = Path(__file__).parents[1] / "assets"
_UI_SETTINGS_SOURCES = [
    (name, value) for name, value in zip(["Swissmedic", "FDA"], SOURCES)
]
# NOTE: make sure that the following dicts is mapping names to the corresponding field names in :class:`Match`
_UI_SETTINGS_RESULTS_COLUMNS = [
    (name, value)
    for name, value in zip(
        GRADIO_RESULTS_COLUMNS, ["sounds_alike", "looks_alike", "combined"]
    )
]


@dataclass
class LocalState:
    """The persistent local state for the app."""

    log_level: str = LOG_LEVEL
    processes: int = PROCESSES


@dataclass
class SessionState:
    """The session dependent state for the app."""

    search: str | None = None
    source: List = field(default_factory=lambda: SOURCES)
    threshold: float = THRESHOLD
    apply_threshold_to: str = GRADIO_RESULTS_COLUMNS_DEFAULT
    is_running: bool = False


class App(BaseApp):
    def __init__(self):
        super().__init__(
            app_name=GRADIO_APP_NAME,
            favicon_path=_ASSETS_PATH / "images" / "bird.png",
            allowed_paths=[str(_ASSETS_PATH.resolve())],
            css_file=_ASSETS_PATH / "css" / "styles.css",
            theme=gr.themes.Soft(),
            local_state=LocalState(),
            session_state=SessionState(),
        )
        self._components: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # User Interface
    # --------------------------------------------------------------------------
    @staticmethod
    def _ui_top_row():
        with gr.Row(elem_classes="top-row"):
            with gr.Column(scale=1):
                gr.Image(
                    value=_ASSETS_PATH / "images" / "logo-circular.png",
                    show_label=False,
                    elem_classes="image",
                )
            with gr.Column(scale=5):
                value = """<div class='top-row title-desc'>
                  <div class='top-row title-desc title'>LASA: Looks Alike Sounds Alike</div>
                  <div class='top-row title-desc desc'><em>A tool to assist experts in identifying similar-looking or sounding drug names</em></div>
                </div>
                """
                gr.Markdown(value=value)

    def _ui_corpus_settings(self):
        """Settings column."""
        with gr.Column(scale=1):
            self._components["settings.source"] = gr.CheckboxGroup(
                label="sources to search",
                show_label=False,
                info="sources to search",
                choices=_UI_SETTINGS_SOURCES,
                value=SOURCES,
                interactive=True,
            )
            self._components["settings.threshold"] = gr.Slider(
                label="threshold value",
                show_label=False,
                info="shreshold value",
                step=1,
                value=THRESHOLD,
                interactive=True,
            )
            self._components["settings.apply_threshold_to"] = gr.Radio(
                label="Threshold Column",
                show_label=False,
                info="apply threshold to",
                choices=_UI_SETTINGS_RESULTS_COLUMNS,
                value=GRADIO_RESULTS_COLUMNS_DEFAULT,
            )

    def _ui_corpus_search_and_results(self):
        """Search and results column."""
        with gr.Column(scale=5):
            with gr.Row(elem_classes="search-container"):
                with gr.Column(scale=3):
                    self._components["search.query"] = gr.Textbox(
                        label="Search term",
                        show_label=False,
                        placeholder="Enter a drug name",
                    )
                with gr.Column(scale=1, elem_classes="search-button"):
                    self._components["search.start"] = gr.HTML(
                        "<div class='search-start'>Search</div>", visible=True
                    )
                    self._components["search.running"] = gr.HTML(
                        "<div class='search-running'>running...</div>", visible=False
                    )

            with gr.Row():
                self._components["search.results"] = gr.Dataframe(
                    value=[],
                    visible=False,
                    column_widths=["36%", "16%", "16%", "16%", "16%"],
                    interactive=False,
                )

    def _ui_corpus_row(self):
        """The main corpus row."""
        with gr.Row(elem_classes="corpus-row"):
            self._ui_corpus_settings()
            self._ui_corpus_search_and_results()

    def setup_ui(self):
        self._ui_top_row()
        self._ui_corpus_row()

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _toggle_button(
        session_state: SessionState,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        session_state.is_running = not session_state.is_running
        if session_state.is_running:
            return session_state, gr.update(visible=False), gr.update(visible=True)
        else:
            return session_state, gr.update(visible=True), gr.update(visible=False)

    @staticmethod
    def _setup_session_state(
        session_state: SessionState,
        source: List[str],
        threshold: float,
        apply_threshold_to: str,
        search: str,
    ) -> SessionState:
        session_state.source = source
        session_state.threshold = threshold
        session_state.apply_threshold_to = apply_threshold_to
        session_state.search = search
        return session_state

    @staticmethod
    def _postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["term", "source", "sounds_alike", "looks_alike", "combined"]].copy()
        float_columns = ["sounds_alike", "looks_alike", "combined"]
        df[float_columns] = df[float_columns].apply(lambda col: (col * 100).round(0))
        df.columns = ["Name", "Source"] + GRADIO_RESULTS_COLUMNS
        return df

    def _get_search_results(
        self, local_state: dict, session_state: SessionState
    ) -> pd.DataFrame:
        # Get matches
        args_ = {
            "search": session_state.search,
            "source": session_state.source,
            "log_level": local_state["log_level"],
        }
        matches = main(args_=args_)
        logger.debug(f"found {len(matches)} matches")
        normalized_threshold = session_state.threshold / 100
        matches = [
            m
            for m in matches
            if getattr(m, session_state.apply_threshold_to) >= normalized_threshold
        ]
        matches.sort(key=lambda m: m.combined, reverse=True)

        # Create and return pandas dataframe
        value = pd.DataFrame([m.to_dict() for m in matches])
        value = self._postprocess_dataframe(df=value)
        logger.debug(
            f"dataframe has shape={value.shape} with threshold={session_state.threshold}"
        )
        return gr.update(value=value, visible=True)

    # --------------------------------------------------------------------------
    # Events
    # --------------------------------------------------------------------------
    def _event_start_search(self):
        gr.on(
            triggers=[
                self._components["search.query"].submit,
                self._components["search.start"].click,
            ],
            fn=self._toggle_button,
            inputs=self._session_state,
            outputs=[
                self._session_state,
                self._components["search.start"],
                self._components["search.running"],
            ],
        ).then(
            fn=self._setup_session_state,
            inputs=[
                self._session_state,
                self._components["settings.source"],
                self._components["settings.threshold"],
                self._components["settings.apply_threshold_to"],
                self._components["search.query"],
            ],
            outputs=self._session_state,
        ).then(
            fn=self._get_search_results,
            inputs=[self._local_state, self._session_state],
            outputs=[self._components["search.results"]],
        ).then(
            fn=self._toggle_button,
            inputs=self._session_state,
            outputs=[
                self._session_state,
                self._components["search.start"],
                self._components["search.running"],
            ],
        )

    def register_events(self):
        self._event_start_search()

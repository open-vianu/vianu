import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
import gradio as gr

from vianu.spock.settings import LOG_LEVEL, N_SCP_TASKS, N_NER_TASKS
from vianu.spock.settings import (
    USE_SCRAPING_SERVICE_FOR,
    SCRAPERAPI_BASE_URL,
    LLM_ENDPOINTS,
    SCRAPING_SOURCES,
    MAX_DOCS,
    MODEL_TEST_QUESTION,
)
from vianu.spock.settings import (
    GRADIO_APP_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_MAX_JOBS,
    GRADIO_UPDATE_INTERVAL,
)
from vianu.spock.src.base import Document, Setup, SpoCK, SpoCKList, QueueItem  # noqa: F401
from vianu import BaseApp
from vianu.spock.__main__ import setup_asyncio_framework
import vianu.spock.app.formatter as fmt
from vianu.spock.src.ner import NERFactory

logger = logging.getLogger(__name__)
load_dotenv()

# App settings
_ASSETS_PATH = Path(__file__).parents[1] / "assets"

_UI_SETTINGS_LLM_ENDPOINT_CHOICES = [
    (name, value) for name, value in zip(["OpenAI", "Ollama"], LLM_ENDPOINTS)
]
if not len(_UI_SETTINGS_LLM_ENDPOINT_CHOICES) == len(LLM_ENDPOINTS):
    raise ValueError(
        "LARGE_LANGUAGE_MODELS and _UI_SETTINGS_LLM_ENDPOINT_CHOICES must have the same length"
    )

_UI_SETTINGS_SOURCE_CHOICES = [
    (name, value)
    for name, value in zip(["PubMed", "EMA", "MHRA", "FDA"], SCRAPING_SOURCES)
]
if not len(_UI_SETTINGS_SOURCE_CHOICES) == len(SCRAPING_SOURCES):
    raise ValueError(
        "SCRAPING_SOURCES and _UI_SETTINGS_SOURCE_CHOICES must have the same length"
    )


@dataclass
class LocalState:
    """The persistent local state."""

    log_level: str = LOG_LEVEL
    n_scp_tasks: int = N_SCP_TASKS
    n_ner_tasks: int = N_NER_TASKS
    max_jobs: int = GRADIO_MAX_JOBS
    update_interval: float = GRADIO_UPDATE_INTERVAL


@dataclass
class SessionState:
    """The session dependent state."""

    # Asyncio setup
    ner_queue: asyncio.Queue | None = None
    scp_tasks: List[asyncio.Task] = field(default_factory=list)
    ner_tasks: List[asyncio.Task] = field(default_factory=list)
    orc_task: asyncio.Task | None = None
    col_task: asyncio.Task | None = None

    # Data
    is_running: bool = False
    spocks: SpoCKList = field(default_factory=list)

    # Scraper settings
    scraperapi_key: str | None = field(
        default_factory=lambda: os.environ.get("SCRAPERAPI_KEY")
    )

    # LLM settings
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    ollama_base_url: str | None = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL")
    )
    connection_is_valid: bool = False

    # Indexes
    _index_running_spock: int | None = None
    _index_active_spock: int | None = None

    def set_running_spock(self, index: int | None) -> None:
        self._index_running_spock = index

    def get_running_spock(self) -> SpoCK:
        return self.spocks[self._index_running_spock]

    def set_active_spock(self, index: int | None) -> None:
        self._index_active_spock = index

    def get_active_spock(self) -> SpoCK:
        return self.spocks[self._index_active_spock]


class App(BaseApp):
    """The main gradio application."""

    def __init__(self):
        super().__init__(
            app_name=GRADIO_APP_NAME,
            favicon_path=_ASSETS_PATH / "images" / "favicon.png",
            allowed_paths=[str(_ASSETS_PATH.resolve())],
            head_file=_ASSETS_PATH / "head" / "scripts.html",
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
                    value=_ASSETS_PATH / "images" / "spock_logo_circular.png",
                    show_label=False,
                    elem_classes="image",
                )
            with gr.Column(scale=5):
                value = """<div class='top-row title-desc'>
                  <div class='top-row title-desc title'>SpoCK: Spotting Clinical Knowledge</div>
                  <div class='top-row title-desc desc'><em>A tool for identifying <b>medicinal products</b> and <b>adverse drug reactions</b> inside publicly available literature</em></div>
                </div>
                """
                gr.Markdown(value=value)

    def _ui_corpus_settings(self):
        """Settings column."""
        with gr.Column(scale=1):
            with gr.Accordion(label="Scraping Endpoint", open=True) as self._components["settings.scraping.accordion"]:
                self._components['settings.scraper_service_usage'] = gr.Checkbox(
                    label="use scraping service",
                    show_label=False,
                    interactive=True,
                )

                # 'scraperapi' specific settings
                with gr.Group(visible=False) as self._components["settings.scraper_service_group"]:
                    value = os.environ.get("SCRAPERAPI_KEY")
                    placeholder = (
                        "api_key of scraperapi" if value is None else None
                    )
                    gr.Markdown("---")
                    self._components["settings.scraperapi_key"] = gr.Textbox(
                        label="api_key",
                        show_label=False,
                        info="api_key",
                        placeholder=placeholder,
                        value=value,
                        interactive=True,
                        type="password",
                    )

            with gr.Accordion(label="LLM Endpoint", open=True) as self._components["settings.llm.accordion"]:
                self._components["settings.llm_radio"] = gr.Radio(
                    label="Model",
                    show_label=False,
                    choices=_UI_SETTINGS_LLM_ENDPOINT_CHOICES,
                    value="openai",
                    interactive=True,
                )

                # 'openai' specific settings
                with gr.Group(visible=True) as self._components[
                    "settings.openai_group"
                ]:
                    value = os.environ.get("OPENAI_API_KEY")
                    placeholder = (
                        "api_key of openai endpoint" if value is None else None
                    )
                    logger.debug(f"openai api_key={value}")
                    gr.Markdown("---")
                    self._components["settings.openai_api_key"] = gr.Textbox(
                        label="api_key",
                        show_label=False,
                        info="api_key",
                        placeholder=placeholder,
                        value=value,
                        interactive=True,
                        type="password",
                    )

                # ollama specific settings
                with gr.Group(visible=False) as self._components[
                    "settings.ollama_group"
                ]:
                    value = os.environ.get("OLLAMA_BASE_URL")
                    placeholder = (
                        "base_url of ollama endpoint" if value is None else None
                    )
                    gr.Markdown("---")
                    self._components["settings.ollama_base_url"] = gr.Textbox(
                        label="base_url",
                        show_label=False,
                        info="base_url",
                        placeholder=placeholder,
                        value=value,
                        interactive=True,
                    )

                self._components["settings.test_connection_button"] = gr.Button(
                    value="Test connection", interactive=True
                )

            with gr.Accordion(label="Filters", open=True, visible=False) as self._components['filters.accordion']:
                self._components["filters.sort_by"] = gr.Radio(
                    label="Sort by",
                    show_label=False,
                    info="sort the results",
                    choices=["source", "#adr"],
                    value="#adr",
                    interactive=True,
                )
                self._components["filters.source"] = gr.CheckboxGroup(
                    label="Sources",
                    show_label=False,
                    info="filter the results",
                    choices=_UI_SETTINGS_SOURCE_CHOICES,
                    interactive=True,
                )
                self._components['filters.selected_adr'] = gr.Dropdown(
                    label="SelectedADR",
                    show_label=False,
                    info="reduce to selected ADR",
                    choices=["adr1", "adr2", "adr3"],
                    value=None,
                    interactive=True,
                    multiselect=True,
                )

    def _ui_corpus_main(self):
        """Search field, job cards, and details."""
        with gr.Column(scale=5):
            # Search text field and start/stop/cancel buttons
            with gr.Row(elem_classes="search-container"):
                with gr.Column(scale=3):
                    self._components["main.search_term"] = gr.Textbox(
                        label="Search",
                        show_label=False,
                        placeholder="Enter your search term",
                    )

                with gr.Column(scale=1, elem_classes="pipeline-button"):
                    self._components["main.start_button"] = gr.HTML(
                        '<div class="button-not-running">Start</div>', visible=True
                    )
                    self._components["main.stop_button"] = gr.HTML(
                        '<div class="button-running">Stop</div>', visible=False
                    )
                    self._components["main.cancel_button"] = gr.HTML(
                        '<div class="canceling">canceling...</div>', visible=False
                    )

            with gr.Row():
                with gr.Accordion(
                    label="Search parameters", open=False
                ) as self._components["settings.parameters"]:
                    self._components["settings.source"] = gr.CheckboxGroup(
                        label="Sources",
                        show_label=False,
                        info="select the sources to scrape",
                        choices=_UI_SETTINGS_SOURCE_CHOICES,
                        value=SCRAPING_SOURCES,
                        interactive=True,
                    )
                    self._components["settings.search_type"] = gr.Radio(
                        label="Search type",
                        show_label=False,
                        info="search type",
                        choices=["fast", "balanced", 'deep'],
                        value="fast",
                        interactive=True,
                    )

            # Job summary cards
            with gr.Row(elem_classes="jobs-container"):
                self._components["main.cards"] = [
                    gr.HTML("", elem_id=f"job-{i}", visible=False)
                    for i in range(GRADIO_MAX_JOBS)
                ]

            # Details of the selected job
            with gr.Row():
                self._components["main.details"] = gr.HTML(
                    '<div class="details-container"></div>'
                )

    def _ui_corpus_row(self):
        """Main corpus with settings, search field, job cards, and details"""
        with gr.Row(elem_classes="bottom-container"):
            self._ui_corpus_settings()
            self._ui_corpus_main()

    def setup_ui(self):
        """Set up the user interface."""
        self._ui_top_row()
        self._ui_corpus_row()
        self._components["timer"] = gr.Timer(
            value=GRADIO_UPDATE_INTERVAL, active=False, render=True
        )

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _show_scraper_service_settings(service_usage: bool, session_state: SessionState) -> Tuple[dict[str, Any], dict[str, Any], SessionState]:
        """Show the settings for the selected scraper service."""
        session_state.use_scraper_service = service_usage
        if service_usage:
            logger.debug("show scraper service settings")
            return gr.update(visible=True), session_state
        else:
            logger.debug("hide scraper service settings")
            return gr.update(visible=False), session_state

    @staticmethod
    def _set_scraperapi_key(api_key: str, session_state: SessionState) -> SessionState:
        """Setup scraperapi api_key"""
        log_key = "*****" if api_key else "None"
        logger.debug(f"set scraperapi api_key {log_key}")
        session_state.scraperapi_key = api_key

    @staticmethod
    def _show_llm_settings(
        endpoint: str, session_state: SessionState
    ) -> Tuple[dict[str, Any], dict[str, Any], SessionState]:
        """Show the settings for the selected LLM model."""
        logger.debug(f"show endpoint={endpoint} settings")
        session_state.connection_is_valid = False
        if endpoint == "ollama":
            return gr.update(visible=True), gr.update(visible=False), session_state
        elif endpoint == "openai":
            return gr.update(visible=False), gr.update(visible=True), session_state
        else:
            return gr.update(visible=False), gr.update(visible=False), session_state

    @staticmethod
    def _set_openai_api_key(api_key: str, session_state: SessionState) -> SessionState:
        """Setup openai api_key"""
        log_key = "*****" if api_key else "None"
        logger.debug(f"set openai api_key {log_key}")
        session_state.openai_api_key = api_key
        session_state.connection_is_valid = False
        return session_state

    @staticmethod
    def _set_ollama_base_url(
        base_url: str, session_state: SessionState
    ) -> SessionState:
        """Setup ollama base_url"""
        logger.debug(f"set ollama base_url={base_url}")
        session_state.ollama_base_url = base_url
        session_state.connection_is_valid = False
        return session_state

    @staticmethod
    async def _test_connection(session_state: SessionState) -> SessionState:
        """Test the connection to the LLM model."""
        setup = session_state.get_running_spock().setup
        endpoint = setup.endpoint
        logger.debug(f"test connection to endpoint={endpoint}")
        try:
            ner = NERFactory.create(setup=setup)
            test_task = asyncio.create_task(ner.test_model_endpoint())
            test_answer = await test_task
            gr.Info(
                f"connection to endpoint={endpoint} is valid: '{MODEL_TEST_QUESTION}' was answered with '{test_answer}'"
            )
            session_state.connection_is_valid = True
        except Exception as e:
            session_state.connection_is_valid = False
            raise gr.Error(f"connection to endpoint={endpoint} failed: {e}")
        return session_state
    
    @staticmethod
    def _get_adr_multiselect_data(
        data: List[Document],
        source: List[str],
        selected_adr: List[str] | None,
    ) -> Tuple[Tuple[List[str], List[str]], List[str]]:
        """Get the adr choices and value for the multiselect field."""

        # Get the ADRs, count them and build the corresponding choices
        data = [d for d in data if d.source in source]
        adrs = [ne.text.upper() for d in data for ne in d.adverse_reactions]
        counter = Counter(adrs)
        counter = sorted(counter.items(), key=lambda x: x[0])
        choices = [(f"{adr} ({count})", adr) for adr, count in counter]

        # Reduce the selected ADRs to the ones that are present in the choices
        if selected_adr is not None and len(selected_adr) > 0:
            poss_vals = [v for _, v in choices]
            value = [a for a in selected_adr if a in poss_vals]
        else:
            value = None
        return choices, value

    def _update_filters(
        self,
        source: List[str],
        selected_adr: List[str] | None,
        session_state: SessionState
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update source and multisleect according to running spock."""
        logger.debug(f"update filters with source={source} and selected_adr={selected_adr}")
        spock = session_state.get_active_spock()
        data = spock.data

        # Update the text and selection of the sources
        src_chc = [
            (f"{txt} ({len([d for d in data if d.source == src])})", src)
            for txt, src in _UI_SETTINGS_SOURCE_CHOICES
        ]
        value = spock.setup.source

        # Update the detected ADRs
        adr_chc, adr_val = self._get_adr_multiselect_data(
            data=data,
            source=source,
            selected_adr=selected_adr,
        )
        return gr.update(choices=src_chc, value=value), gr.update(choices=adr_chc, value=adr_val)

    def _update_adr_multiselect(
        self,
        source: List[str],
        selecte_adr: List[str] | None,
        session_state: SessionState
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = session_state.get_active_spock().data
        choices, value = self._get_adr_multiselect_data(
            data=data,
            source=source,
            selected_adr=selecte_adr,
        )
        return gr.update(choices=choices, value=value)

    @staticmethod
    def _feed_cards_to_ui(
        local_state: dict, session_state: SessionState
    ) -> List[dict[str, Any]]:
        """For all existing SpoCKs, create and feed the job cards to the UI."""
        spocks = session_state.spocks
        logger.debug(f"feeding cards to UI (len(spocks)={len(spocks)})")

        # Create the job cards for the existing spocks
        cds = []
        for i, spk in enumerate(spocks):
            html = fmt.get_job_card_html(i, spk)
            cds.append(gr.update(value=html, visible=True))

        # Extdend with not-visible cards
        # Note: for gradio >= 5.0.0 this logic could be replaces with dynamic number of gr.Blocks
        # (see https://www.gradio.app/guides/dynamic-apps-with-render-decorator)
        cds.extend(
            [
                gr.update(visible=False)
                for _ in range(local_state["max_jobs"] - len(spocks))
            ]
        )
        return cds

    @staticmethod
    def _feed_details_to_ui(
        session_state: SessionState,
        sort_by: str = "#adr",
        source: List[str] = SCRAPING_SOURCES,
        selected_adr: List[str] | None = None,
    ) -> str:
        """Collect the html texts for the documents of the selected job and feed them to the UI."""
        if len(session_state.spocks) == 0:
            return fmt.get_details_html([])

        # Select and filter the data
        data = session_state.get_active_spock().data
        data = [d for d in data if d.source in source]
        if selected_adr is not None and len(selected_adr) > 0:
            data = [d for d in data if any([ne.text.upper() in selected_adr for ne in d.adverse_reactions])]

        logger.debug(f"feeding details to UI (len(data)={len(data)})")
        return fmt.get_details_html(data, sort_by=sort_by, source=source)

    async def _check_settings(self, service_usage: bool, session_state: SessionState) -> SessionState:
        """Check if the settings correct."""
        setup = session_state.get_running_spock().setup
        
        # Check the scraper settings
        if service_usage:
            if session_state.scraperapi_key is None:
                raise gr.Error("scraping service' api_key is not set")

        # Check the LLM settings
        endpoint = setup.endpoint
        if endpoint == "openai" and session_state.openai_api_key is None:
            raise gr.Error("OpenAI api_key is not set")
        elif endpoint == "ollama" and session_state.ollama_base_url is None:
            raise gr.Error("Ollama base_url is not set")

        # Check connection to the model
        if not session_state.connection_is_valid:
            session_state = await self._test_connection(session_state=session_state)
        return session_state

    @staticmethod
    def _toggle_button(
        session_state: SessionState,
    ) -> Tuple[SessionState, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Toggle the state of the pipleline between running <-> not_running.

        As a result the corresponding buttons (Start, Stop, canceling...) are shown/hidden.
        """
        logger.debug(
            f"toggle button (is_running={session_state.is_running}->{not session_state.is_running})"
        )
        session_state.is_running = not session_state.is_running
        if session_state.is_running:
            # Show the stop button and hide the start/cancel button
            return (
                session_state,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            # Show the start button and hide the stop/cancel button
            return (
                session_state,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    @staticmethod
    def _set_accordion_visibilities() -> Dict[str, Any]:
        """Set open/close of accordions:
            settings.parameters: False
            settings.scraping.accordion: False
            settings.llm.accordion: False
            filters.accordion: True
        """
        return (
            gr.update(open=False),
            gr.update(open=False),
            gr.update(open=False),
            gr.update(visible=True, open=True),
        )

    @staticmethod
    def _show_cancel_button() -> Tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Shows the cancel button and hides the start and stop button."""
        logger.debug("show cancel button")
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    @staticmethod
    def _setup_spock(
        term: str,
        service_usage: bool,
        endpoint: str,
        source: List[str],
        search_type: str,
        local_state: dict,
        session_state: SessionState,
    ) -> SessionState:
        """Setup a new SpoCK object."""
        max_jobs = local_state["max_jobs"]
        spocks = session_state.spocks

        # Check if the maximum number of jobs is reached and pop the last job if necessary
        if len(spocks) >= max_jobs:
            msg = f'max number of jobs ({max_jobs}); last job "{spocks[-1].setup.term}" is removed'
            gr.Warning(msg)
            logger.warning(msg)
            spocks.pop(-1)

        # Setup the running_spock and append it to the list of spocks
        msg = f'started SpoCK for "{term}"'
        gr.Info(msg)
        logger.info(msg)

        # Create new SpoCK object
        max_docs_src = int(MAX_DOCS[search_type] / len(source))
        setup = Setup(
            id_=f"{term} {source} {endpoint}",
            term=term,
            service=service_usage,
            endpoint=endpoint,
            source=source,
            max_docs_src=max_docs_src,
            log_level=local_state["log_level"],
            n_scp_tasks=local_state["n_scp_tasks"],
            n_ner_tasks=local_state["n_ner_tasks"],
            scraperapi_key=session_state.scraperapi_key,
            openai_api_key=session_state.openai_api_key,
            llama_base_url=session_state.ollama_base_url,
        )
        spock = SpoCK(
            id_=setup.id_,
            status="running",
            setup=setup,
            started_at=setup.submission,
            data=[],
        )

        # Set the running and focused SpoCK to be the new SpoCK object
        index = 0
        spocks.insert(index, spock)
        session_state.set_running_spock(index=index)
        session_state.set_active_spock(index=index)
        return session_state

    @staticmethod
    async def _collector(session_state: SessionState) -> None:
        """Append the processed document(s) from the `session_state.ner_queue` to the running spock."""
        running_spock = session_state.get_running_spock()
        ner_queue = session_state.ner_queue
        logger.debug(f"starting collector (term={running_spock.setup.term})")

        while True:
            item = await ner_queue.get()  # type: QueueItem
            # Check stopping condition (added by the `orchestrator` in `vianu.spock.__main__`)
            if item is None:
                ner_queue.task_done()
                break
            running_spock.data.append(item.doc)
            ner_queue.task_done()

    async def _setup_asyncio_framework(
        self, service_usage: bool, session_state: SessionState
    ) -> SessionState:
        """ "Start the SpoCK processes by setting up the asyncio framework and starting the asyncio tasks.

        Main components of asyncio framework are:
        - ner_queue: queue for collecting results from named entity recognition tasks
        - scp_tasks: scraping tasks (cf. `vianu.spock.src.scp`)
        - ner_tasks: named entity recognition tasks (cf. `vianu.spock.src.ner`)
        - orc_task: orchestrating the process
        - col_task: collect and assemble the final results
        """
        logger.info("setting up asyncio framework")

        # Setup asyncio tasks as in `vianu.spock.__main__`
        setup = session_state.get_running_spock().setup
        ner_queue, scp_tasks, ner_tasks, orc_task = setup_asyncio_framework(setup=setup)
        session_state.ner_queue = ner_queue
        session_state.scp_tasks = scp_tasks
        session_state.ner_tasks = ner_tasks
        session_state.orc_task = orc_task

        # Setup the app specific collection task
        col_task = asyncio.create_task(self._collector(session_state=session_state))
        session_state.col_task = col_task

        return session_state

    @staticmethod
    async def _conclusion(session_state: SessionState) -> SessionState:
        # Wait collector task to finish and join ner_queue
        try:
            await session_state.col_task
        except asyncio.CancelledError:
            logger.warning("collector task canceled")
            return session_state  # This stops the _conclusion step in the case the _canceling step was triggered
        except Exception as e:
            logger.error(f"collector task failed with error: {e}")
            raise e
        await session_state.ner_queue.join()

        # Update the running_spock with the final data
        running_spock = session_state.get_running_spock()
        running_spock.status = "completed"
        running_spock.finished_at = datetime.now()

        # Log the conclusion and update/empty the running_spock
        gr.Info(f'job "{running_spock.setup.term}" finished')
        logger.info(
            f'job "{running_spock.setup.term}" finished in {running_spock.runtime()}'
        )

        return session_state

    @staticmethod
    async def _canceling(session_state: SessionState) -> SessionState:
        """Cancel all running :class:`asyncio.Task`."""
        running_spock = session_state.get_running_spock()
        gr.Info(f'canceled SpoCK for "{running_spock.setup.term}"')

        # Update the running_spock
        running_spock.status = "stopped"
        running_spock.finished_at = datetime.now()

        # Cancel scraping tasks
        logger.warning("canceling scraping tasks")
        for task in session_state.scp_tasks:
            task.cancel()
        await asyncio.gather(*session_state.scp_tasks, return_exceptions=True)

        # Cancel named entity recognition tasks
        logger.warning("canceling named entity recognition tasks")
        for task in session_state.ner_tasks:
            task.cancel()
        await asyncio.gather(*session_state.ner_tasks, return_exceptions=True)

        # Cancel orchestrator task
        logger.warning("canceling orchestrator task")
        session_state.orc_task.cancel()
        await asyncio.gather(
            session_state.orc_task, return_exceptions=True
        )  # we use return_exceptions=True to avoid raising exceptions due to the subtasks being allready canceled`

        # Cancel collector task
        logger.warning("canceling collector task")
        session_state.col_task.cancel()
        await asyncio.gather(
            session_state.col_task, return_exceptions=True
        )  # see remark above

        return session_state

    @staticmethod
    def _change_active_spock_number(
        session_state: SessionState, index: int
    ) -> SessionState:
        logger.debug(f"card clicked={index}")
        session_state.set_active_spock(index=index)
        return session_state

    # --------------------------------------------------------------------------
    # Events
    # --------------------------------------------------------------------------
    def _event_timer(self):
        self._components["timer"].tick(
            fn=self._update_filters,
            inputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
                self._session_state,
            ],
            outputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
        ).then(
            fn=self._feed_cards_to_ui,
            inputs=[self._local_state, self._session_state],
            outputs=self._components["main.cards"],
        ).then(
            fn=self._feed_details_to_ui,
            inputs=[
                self._session_state,
                self._components["filters.sort_by"],
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            outputs=self._components["main.details"],
        )
    
    def _event_use_scraper_service(self):
        """Use scraper service and show the corresponding settings."""
        self._components["settings.scraper_service_usage"].change(
            fn=self._show_scraper_service_settings,
            inputs=[
                self._components["settings.scraper_service_usage"],
                self._session_state,
            ],
            outputs=[
                self._components["settings.scraper_service_group"],
                self._session_state,
            ],
        )
    
    def _event_settings_scraperapi(self):
        """Callback of the scraperapi settings."""
        self._components["settings.scraperapi_key"].change(
            fn=self._set_scraperapi_key,
            inputs=[self._components["settings.scraperapi_key"], self._session_state],
            outputs=self._session_state,
        )

    def _event_choose_llm(self):
        """Choose LLM model show the correspoding settings."""
        self._components["settings.llm_radio"].change(
            fn=self._show_llm_settings,
            inputs=[
                self._components["settings.llm_radio"],
                self._session_state,
            ],
            outputs=[
                self._components["settings.ollama_group"],
                self._components["settings.openai_group"],
                self._session_state,
            ],
        )

    def _event_settings_openai(self):
        """Callback of the openai settings."""
        self._components["settings.openai_api_key"].change(
            fn=self._set_openai_api_key,
            inputs=[self._components["settings.openai_api_key"], self._session_state],
            outputs=self._session_state,
        )

    def _event_settings_ollama(self):
        """Callback of the ollama settings."""
        self._components["settings.ollama_base_url"].change(
            fn=self._set_ollama_base_url,
            inputs=[self._components["settings.ollama_base_url"], self._session_state],
            outputs=self._session_state,
        )

    def _event_test_connection(self):
        """Test the connection to the LLM model."""
        self._components["settings.test_connection_button"].click(
            fn=self._test_connection,
            inputs=self._session_state,
            outputs=self._session_state,
        )

    def _event_filters(self):
        self._components["filters.sort_by"].change(
            fn=self._feed_details_to_ui,
            inputs=[
                self._session_state,
                self._components["filters.sort_by"],
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            outputs=self._components["main.details"],
        )
        self._components["filters.source"].change(
            fn=self._update_adr_multiselect,
            inputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
                self._session_state,
            ],
            outputs=[
                self._components["filters.selected_adr"],
            ],
        ).then(
            fn=self._feed_details_to_ui,
            inputs=[
                self._session_state,
                self._components["filters.sort_by"],
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            outputs=self._components["main.details"],
        )
        self._components["filters.selected_adr"].change(
            fn=self._feed_details_to_ui,
            inputs=[
                self._session_state,
                self._components["filters.sort_by"],
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            outputs=self._components["main.details"],
        )

    def _event_start_spock(self) -> None:
        search_term = self._components["main.search_term"]
        start_button = self._components["main.start_button"]
        timer = self._components["timer"]

        gr.on(
            triggers=[search_term.submit, start_button.click],
            # Setup the running spock for having the :class:`Setup` object
            fn=self._setup_spock,   
            inputs=[
                search_term,
                self._components["settings.scraper_service_usage"],
                self._components["settings.llm_radio"],
                self._components["settings.source"],
                self._components["settings.search_type"],
                self._local_state,
                self._session_state,
            ],
            outputs=self._session_state,
        ).then(
            fn=self._check_settings,
            inputs=[
                self._components["settings.scraper_service_usage"],
                self._session_state,
            ],
            outputs=self._session_state,
        ).success(
            fn=self._toggle_button,
            inputs=self._session_state,
            outputs=[
                self._session_state,
                self._components["main.start_button"],
                self._components["main.stop_button"],
                self._components["main.cancel_button"],
            ],
        ).then(
            fn=self._set_accordion_visibilities,
            outputs=[
                self._components["settings.parameters"],
                self._components['settings.scraping.accordion'],
                self._components['settings.llm.accordion'],
                self._components['filters.accordion'],
            ],
        ).then(
            fn=self._setup_asyncio_framework,
            inputs=[
                self._components["settings.scraper_service_usage"],
                self._session_state,
            ],
            outputs=self._session_state,
        ).then(
            # Empty the search term in the UI
            fn=lambda: None,
            outputs=search_term,  
        ).then(
            fn=self._update_filters,
            inputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
                self._session_state,
            ],
            outputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
        ).then(
            fn=self._feed_cards_to_ui,
            inputs=[self._local_state, self._session_state],
            outputs=self._components["main.cards"],
        ).then(fn=lambda: gr.update(active=True), outputs=timer).then(
            fn=self._conclusion,
            inputs=self._session_state,
            outputs=self._session_state,
        ).then(
            # filters, cards and ui is updated one more time
            # in order to not depend on the state of the timer
            fn=self._update_filters,
            inputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
                self._session_state,
            ],
            outputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
        ).then(
            fn=self._feed_cards_to_ui,
            inputs=[self._local_state, self._session_state],
            outputs=self._components["main.cards"],
        ).then(
            fn=self._feed_details_to_ui,  
            inputs=[
                self._session_state,
                self._components["filters.sort_by"],
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            outputs=self._components["main.details"],
        ).then(
            # Turn off the timer
            fn=lambda: gr.update(active=False), outputs=timer
        ).then(
            fn=self._toggle_button,
            inputs=self._session_state,
            outputs=[
                self._session_state,
                self._components["main.start_button"],
                self._components["main.stop_button"],
                self._components["main.cancel_button"],
            ],
        )

    def _event_stop_spock(self):
        # NOTE: when `stop_button.click` is triggered, the above pipeline (started by `search_term.submit` or
        # `start_button.click`) is still running and awaiting the `_conclusion` step to finish. The `stop_button.click`
        # event will cause the `_conclusion` step to terminate, after which the subsequent steps will still be executed;
        # -> therefore, there is no need to add these steps here.
        self._components["main.stop_button"].click(
            fn=self._show_cancel_button,
            outputs=[
                self._components["main.start_button"],
                self._components["main.stop_button"],
                self._components["main.cancel_button"],
            ],
        ).then(
            fn=self._canceling,
            inputs=self._session_state,
            outputs=self._session_state,
        )

    def _event_card_click(self):
        for index, crd in enumerate(self._components["main.cards"]):
            crd.click(
                fn=self._change_active_spock_number,
                inputs=[self._session_state, gr.Number(value=index, visible=False)],
                outputs=self._session_state,
            ).then(
                fn=self._update_filters,
                inputs=[
                    self._components["filters.source"],
                    self._components["filters.selected_adr"],
                    self._session_state,
                ],
                outputs=[
                self._components["filters.source"],
                self._components["filters.selected_adr"],
            ],
            ).then(
                fn=self._feed_details_to_ui,
                inputs=[
                    self._session_state,
                    self._components["filters.sort_by"],
                    self._components["filters.source"],
                    self._components["filters.selected_adr"],
                ],
                outputs=self._components["main.details"],
            )

    def register_events(self):
        """Register the events."""
        # Setup timer for feed cards and details
        self._event_timer()

        # Settings events
        self._event_use_scraper_service()
        self._event_settings_scraperapi()
        self._event_choose_llm()
        self._event_settings_ollama()
        self._event_settings_openai()
        self._event_test_connection()

        # Filter events
        self._event_filters()

        # Start/Stop events
        self._event_start_spock()
        self._event_stop_spock()

        # Card click events for showing details
        self._event_card_click()


if __name__ == "__main__":
    from vianu.spock.app import App

    app = App()
    demo = app.make()
    demo.queue().launch(
        favicon_path=app.favicon_path,
        inbrowser=True,
        allowed_paths=[
            str(_ASSETS_PATH.resolve()),
        ],
        server_port=GRADIO_SERVER_PORT,
    )

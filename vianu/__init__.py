from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import gradio as gr
from gradio.events import Dependency

LOG_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"


class BaseApp(ABC):
    """The abstract base class of the main gradio application."""

    def __init__(
        self,
        app_name: str | None = None,
        favicon_path: Path | None = None,
        allowed_paths: List[str] | None = None,
        head_file: Path | None = None,
        css_file: Path | None = None,
        theme: gr.Theme | None = None,
        local_state: Any | None = None,
        session_state: Any | None = None,
    ):
        """
        Args:
            app_name: The name of the application. Defaults to None.
            favicon_path: The favicon file as a :class:`pathlib.Path`. Defaults to None.
            head_file: Custom html code as a :class:`pathlib.Path` to a html file. Defaults to None.
            css_file (Path, optional): Custom css as a :class:`pathlib.Path` to a css file. Defaults to None.
            theme (gr.Theme, optional): The theme of the application. Defaults to None.
            local_state (Any, optional): The local state, where data persists in the browser's localStorage even after the page is refreshed or closed. Should be a json-serializable value (accessible only through it's serialized form). Defaults to None.
            session_state (Any, optional): The session state, where data persists across multiple submits within a page session. Defaults to None
        """
        self.favicon_path = favicon_path
        self.allowed_paths = allowed_paths

        self._app_name = app_name
        self._head_file = head_file
        self._css_file = css_file
        self._theme = theme

        self._local_state = gr.BrowserState(local_state)
        self._session_state = gr.State(session_state)

    @abstractmethod
    def setup_ui(self):
        """Set up the user interface."""
        pass

    @abstractmethod
    def register_events(self):
        """Register the events."""
        pass

    def make(self) -> Dependency:
        with gr.Blocks(
            title=self._app_name,
            head_paths=self._head_file,
            css_paths=self._css_file,
            theme=self._theme,
        ) as demo:
            self._local_state.render()
            self._session_state.render()
            self.setup_ui()
            self.register_events()

            demo.load()
        return demo

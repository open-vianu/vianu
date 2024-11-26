import logging
import os
from pathlib import Path

from vianu.spock.app import App
from vianu.spock.settings import LOG_LEVEL, LOG_FMT, GRADIO_SERVER_PORT

logging.basicConfig(level=LOG_LEVEL.upper(), format=LOG_FMT)
os.environ["GRADIO_SERVER_PORT"] = str(GRADIO_SERVER_PORT)

if __name__ == "__main__":
    app = App()
    demo = app.make()
    demo.queue().launch(
        favicon_path=app.favicon_path,
        inbrowser=True,
        allowed_paths=app.allowed_paths,
    )

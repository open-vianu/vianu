import logging
import os

from vianu import LOG_FMT
from vianu.lasa.settings import LOG_LEVEL, GRADIO_SERVER_PORT
from vianu.lasa.app import App

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

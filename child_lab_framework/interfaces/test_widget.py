import asyncio
from websockets.sync.client import connect

from child_lab_framework.docker import run

NAME = "test-widget"
ADDRESS = "127.0.0.1"
PORT = 15_001

class TestWidget():

    def __init__(self):
        self.__container_handle = run(NAME, PORT, ADDRESS)

    def __call__(self):
        from time import sleep
        sleep(5)

        with connect(f"ws://{ADDRESS}:{PORT}") as ws:
            ws.send("Hello world!")
            message: str = ws.recv()
            return message
    
    def __del__(self):
        self.__container_handle()

if __name__ == "__main__":
    mock_model = TestWidget()

    print(mock_model())

    from time import sleep
    sleep(2)

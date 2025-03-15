import logging
import threading
import os
import json
import sys
from datetime import datetime

from logic.config import ConfigManager
from logic.Transcriptor import Transcriptor, SoundEvent

from aiohttp import web
from aiohttp.web_runner import GracefulExit
from aiohttp_sse import sse_response
import asyncio
from pynput import keyboard

logging.basicConfig(filename="app.log", filemode="w", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
recording_logic_thread = None
required_keys = {keyboard.Key.alt, keyboard.KeyCode.from_char('r'.lower())}
ending_keys = {keyboard.Key.ctrl, keyboard.Key.esc}
pressed_keys = set()


def update_multi_key_status():
    global global_is_hotkey_pressed_flag
    global recording_logic_thread
    if required_keys.issubset(pressed_keys):
        if (recording_logic_thread is None or not recording_logic_thread.is_alive()) and not transcriptor.is_hotkey_pressed_flag:
            transcriptor.is_hotkey_pressed_flag = True
            recording_logic_thread = threading.Thread(target=transcriptor.main_logic, args=())
            recording_logic_thread.start()
        else:
            log.debug("Still recording")
        log.debug("Is recording")
    else:
        transcriptor.is_hotkey_pressed_flag = False
        log.debug("Is not recording")


def on_press(key):
    if not pressed_keys.__contains__(key):
        pressed_keys.add(key)
        update_multi_key_status()


def on_release(key):
    if ending_keys.issubset(pressed_keys):
        return False
    pressed_keys.discard(key)
    update_multi_key_status()

async def event_stream(request: web.Request) -> web.StreamResponse:
    global input
    async with sse_response(request) as resp:
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader(loop=loop)
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while resp.is_connected():
            print("before read", flush=True)
            input = await reader.read()
            print(f"read {input}", flush=True)

            if input == TRANSCRIPTION_EOS:
                raise GracefulExit

            if TRANSCRIPTION_SEPARATOR and TRANSCRIPTION_SEPARATOR in input:
                split = input.split(TRANSCRIPTION_SEPARATOR)
                confirmed = split[0]
                potential = split[1]
            else:
                confirmed = input
                potential = ""

            if TRANSCRIPTION_PROPAGATE:
                print(input)

            eol = input == TRANSCRIPTION_EOT
            data = json.dumps({"confirmed": f"{confirmed}", "potential": f"{potential}", "time": f"{datetime.now()}", "eol": f"{eol}"})
            await asyncio.sleep(PROCESSING_DELAY)
            await resp.send(data)
    return resp


async def index(request):
    return web.FileResponse('index.html')


if __name__ == '__main__':
    app = web.Application()
    app.router.add_get("/events", event_stream)
    app.router.add_get("/", index)
    web.run_app(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.load_from_file("./config.yaml")

    transcriptor = Transcriptor(sound_dir, config_manager.get_config())

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except Exception as e:
        log.error(e)
    transcriptor.play_sound_async(SoundEvent.PROCESSING_END)
    # if TRANSCRIPTION_TO_TERMINAL_SHARE_EOS:
    #     TRANSCRIPTION_TO_TERMINAL = True
    #     to_terminal(TRANSCRIPTION_TO_TERMINAL_EOS)

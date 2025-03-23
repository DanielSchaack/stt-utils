import logging
import threading
import os
import json
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
eos = False


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
    global eos
    if ending_keys.issubset(pressed_keys):
        print(transcriptor.config.transcription.terminal_eos)
        eos = True
        return False
    pressed_keys.discard(key)
    update_multi_key_status()


# TODO refactor event_stream for using this instead of pulling
async def get_latest_item(queue: asyncio.Queue):
    while not queue.empty():
        item = await queue.get()
        queue.task_done()
    return item


async def event_stream(request: web.Request) -> web.StreamResponse:
    global eos
    confirmed: str = ""
    potential: str = ""
    async with sse_response(request) as resp:
        while resp.is_connected():
            if eos:
                raise GracefulExit

            # could be handled through events, or async queues or whatever, but am lazy
            cur_confirmed = transcriptor.confirmed
            cur_potential = transcriptor.potential
            if confirmed != cur_confirmed or potential != cur_potential:
                confirmed = cur_confirmed
                potential = cur_potential
            else:
                await asyncio.sleep(config_manager.config.processing.delay)
                continue

            eot = transcriptor.eot
            data = json.dumps({"confirmed": f"{confirmed}", "potential": f"{potential}", "time": f"{datetime.now()}", "eot": f"{eot}"})
            await resp.send(data)
    return resp


async def index(request):
    return web.FileResponse('index.html')


def run_app():
    app = web.Application()
    app.router.add_get("/events", event_stream)
    app.router.add_get("/", index)
    web.run_app(app=app, host="127.0.0.1", port=42069)


if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.load_from_file("./config.yaml")

    transcriptor = Transcriptor(sound_dir, config_manager.get_config())

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            if config_manager.config.ui.enabled:
                run_app()
            else:
                listener.join()
    except Exception as e:
        log.error(e)
    finally:
        transcriptor.play_sound_async(SoundEvent.PROCESSING_END)

        if config_manager.config.transcription.terminal_share_eos:
            config_manager.config.transcription.to_terminal = True
            transcriptor.update_config(config_manager.get_config())
            transcriptor.to_terminal(config_manager.config.transcription.terminal_eos)

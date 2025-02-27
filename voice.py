import time
import threading
import queue
import logging
import math
import os
import gc
import unicodedata

from typing import Optional
from dataclasses import dataclass
from enum import Enum

from faster_whisper import WhisperModel
import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf

logging.basicConfig(filename="app.log", filemode="w", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

RECORDING_CHANNELS = 1
RECORDING_RATE = 16000
RECORDING_CHUNK_SIZE = RECORDING_RATE
RECORDING_DEBUG_OUTPUT = False

TRANSCRIPTION_LANGUAGE = "de"
TRANSCRIPTION_CHUNK_STEP_SIZE = 1
TRANSCRIPTION_MAX_TIMESPAN = 10
TRANSCRIPTION_TO_CLIPBOARD = True
TRANSCRIPTION_TO_CONSOLE = True

PROCESSING_DELAY = 0.1
PROCESSING_STOP_TIMESPAN_DONE = 5
PROCESSING_MIN_TIMESPAN_DONE = 2
PROCESSING_MIN_DUPE_WORD_COUNT = 2
PROCESSING_MIN_DUPE_BETWEEN_RECORDS_NEEDED = 2

SOUND_RECORDING_START_ACTIVE = True
SOUND_RECORDING_END_ACTIVE = True
SOUND_PROCESSING_END_ACTIVE = True
SOUND_RELATIVE_LENGTH = 1.0
SOUND_RELATIVE_VOLUME = 1.0
SOUND_RELATIVE_SPEED = 2.0

MODEL_NAME = "turbo"
MODEL_DEVICE = "cuda"  # cuda or cpu
MODEL_COMPUTE_TYPE = "int8"   # float16 or int8
MODEL_DIR = "./models"

q = queue.Queue()
sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
is_recording = False


class SoundEvent(Enum):
    RECORDING_START = 1
    RECORDING_END = 2
    PROCESSING_END = 3


@dataclass
class Word:
    start: np.float64
    end: np.float64
    probability: np.float64
    word: str


def callback_send_to_queue(indata, frames, time, status):
    log.debug(f"Received {len(indata)} bytes of audio data")
    q.put(indata.copy())
    return None


def get_audio_window(total_data, window_start_step: float, window_stop_step: int, chunksize: int) -> np.ndarray:
    """Extract a window of audio data and convert to numpy array."""
    start_index = math.floor(window_start_step * chunksize)
    end_index = window_stop_step * chunksize

    # Due to required int16, ensure difference is a multiple of 2
    if (end_index - start_index) % 2 == 1:
        start_index -= 1

    log.debug(f"Start index: {start_index}, End index: {end_index}")
    return (np.frombuffer(total_data[start_index:end_index], np.int16)
            .flatten()
            .astype(np.float32) / 32768.0
            )


def transcribe_window(model: WhisperModel, np_data: np.ndarray, context: str) -> tuple[str, list[Word]]:
    """Transcribe audio data using the Whisper model."""
    segments, _ = model.transcribe(audio=np_data,
                                   language=TRANSCRIPTION_LANGUAGE,
                                   initial_prompt=context if context != "" else "",
                                   beam_size=5,
                                   without_timestamps=False,
                                   word_timestamps=True,
                                   condition_on_previous_text=False,
                                   vad_filter=True,
                                   vad_parameters=dict(min_silence_duration_ms=1000)
                                   )

    all_words = []
    for segment in segments:
        all_words.extend(segment.words)

    log.debug(f"Described with context '{context} the words: {all_words}")
    transcription = "".join(word.word for word in all_words)
    return transcription, all_words


def find_duplicate_words(list_of_lists: list[list[Word]]) -> list[tuple[list[int], list[int]]]:
    """Find duplicates between different word lists."""
    max_dupes = []
    len_lists = len(list_of_lists)

    for i in range(len_lists):
        current_dupe = []
        compared_to = []

        for j in range(i + 1, len_lists):
            if j < len_lists:
                amount_dupe, dupe_words = compare_word_lists(list_of_lists[i], list_of_lists[j])
                log.debug(f"Amount dupes: {amount_dupe} with duplicate words: {dupe_words}")

                if amount_dupe >= PROCESSING_MIN_DUPE_WORD_COUNT:
                    compared_to.append(j)
                    current_dupe.append(amount_dupe)

        max_dupes.append((current_dupe, compared_to))
    log.debug(f"Calulated duplicate words: {max_dupes}")
    return max_dupes


def normalize_string(s):
    return unicodedata.normalize('NFC', s.strip()).casefold()


def compare_word_lists(list_a: list[Word], list_b: list[Word]) -> tuple[int, list[str]]:
    """Compare two word lists and count matching words."""
    min_len = min(len(list_a), len(list_b))
    duplicate = 0
    words = []

    for i in range(min_len):
        word_a = normalize_string(list_a[i].word)
        word_b = normalize_string(list_b[i].word)
        log.debug(f"Comparing {i}th word '{word_a}' with '{word_b}'")
        if word_a == word_b:
            duplicate += 1
            words.append(list_a[i].word)
        else:
            return duplicate, words

    return duplicate, words


def get_index_dupes(dupes_list: list[int], index_list: list[int]) -> tuple[int, int]:
    """Find the index of the minimum duplication count."""
    min_dupes: int = min(dupes_list)
    index: int = len(index_list) - 1

    for current_dupe in reversed(dupes_list):
        if current_dupe == min_dupes:
            break
        index -= 1

    return index, min_dupes


def get_confirmed_potential_words(dupe_lists: list[tuple[list[int], list[int]]],
                                  word_lists: list[list[Word]]) -> tuple[Optional[list[Word]], Optional[list[Word]]]:
    """Determine which words are confirmed and which are potentially available."""
    index = len(word_lists) - 1

    for dupe_list in reversed(dupe_lists):
        list_amount_dupes, list_dupes_to = dupe_list
        log.debug(f"List of duplicate amounts: {list_amount_dupes}")
        log.debug(f"List mapping duplicates to words: {list_dupes_to}")

        if len(list_amount_dupes) >= PROCESSING_MIN_DUPE_BETWEEN_RECORDS_NEEDED:
            min_index, min_value = get_index_dupes(list_amount_dupes, list_dupes_to)
            return word_lists[index][:min_value], word_lists[index][min_value:]
        index -= 1
    return None, None


def join_words(word_list: list[Word]) -> str:
    """Join a list of words into a string."""
    return "".join(word.word for word in word_list)


def to_clipboard(text: str):
    if TRANSCRIPTION_TO_CLIPBOARD:
        pyperclip.copy(text.strip())


def to_console(text: str):
    if TRANSCRIPTION_TO_CONSOLE:
        print(text.strip())


def play_sound(sound: SoundEvent):
    if sound == SoundEvent.RECORDING_START and SOUND_RECORDING_START_ACTIVE:
        file_path = sound_dir + "/start_recording.wav"
    elif sound == SoundEvent.RECORDING_END and SOUND_RECORDING_END_ACTIVE:
        file_path = sound_dir + "/end_recording.wav"
    elif sound == SoundEvent.PROCESSING_END and SOUND_PROCESSING_END_ACTIVE:
        file_path = sound_dir + "/end_processing.wav"
    else:
        return

    try:
        data, samplerate = sf.read(file_path, dtype='float32')
        audiolength = math.ceil(len(data) * SOUND_RELATIVE_LENGTH)
        audio = data[:audiolength - 1]
        audio *= SOUND_RELATIVE_VOLUME
        samplerate *= SOUND_RELATIVE_SPEED
        sd.play(audio, samplerate)
        sd.wait()
    except Exception as e:
        log.error(e)


def producer():
    """Record audio and add it to the queue."""
    global is_recording

    play_sound(SoundEvent.RECORDING_START)
    log.info("Starting audio recording...")

    stream = sd.InputStream(
        samplerate=RECORDING_RATE,
        blocksize=RECORDING_CHUNK_SIZE,
        channels=RECORDING_CHANNELS,
        dtype='int16',
        callback=callback_send_to_queue
    )

    with stream:
        is_recording = True
        while is_recording:
            time.sleep(PROCESSING_DELAY)
        stream.close()

    play_sound(SoundEvent.RECORDING_END)
    log.info("End of producer thread")


def consumer():
    """Process audio data from the queue and transcribe it."""
    log.info("Consumer thread started")

    total_data = np.array([], dtype='int16')
    window_start_step = 0
    window_stop_step = TRANSCRIPTION_CHUNK_STEP_SIZE
    available_chunks = 0
    transcribed = ""
    confirmed_transcribed = ""
    word_lists = []
    do_run = True

    while do_run:
        if window_stop_step - math.ceil(window_start_step) > PROCESSING_STOP_TIMESPAN_DONE and not transcribed.strip():
            log.info(f"Confirmed: {confirmed_transcribed} | Potential: {transcribed}")
            do_run = False

        if q.qsize() != 0:
            total_data = np.append(total_data, q.get())
            q.task_done()
        else:
            time.sleep(PROCESSING_DELAY)
            continue

        available_chunks = len(total_data) // RECORDING_CHUNK_SIZE
        log.debug(f"Available chunks: {available_chunks}, current size: {len(total_data)}, current start step: {window_start_step}, current stop step: {window_stop_step}")

        # Process available audio data
        if window_stop_step <= available_chunks:
            np_data = get_audio_window(total_data, window_start_step, window_stop_step, RECORDING_CHUNK_SIZE)
            transcribed, all_words = transcribe_window(whisper, np_data, confirmed_transcribed)
            log.info(f"From {window_start_step:.2f} seconds to {window_stop_step:.2f} seconds transcribed to: {transcribed}")

            if len(all_words) > 0:
                word_lists.append(all_words)

            if window_stop_step - math.ceil(window_start_step) >= TRANSCRIPTION_MAX_TIMESPAN:
                log.info(f"Reached max timespan {TRANSCRIPTION_MAX_TIMESPAN}, evaluating current words to move on")
                confirmed_transcribed += transcribed
                transcribed = ""
                last_confirmed_word = all_words[-1]
                window_start_step += last_confirmed_word.end
                window_stop_step = math.ceil(window_start_step + TRANSCRIPTION_CHUNK_STEP_SIZE)
                word_lists.clear()

                continue
            else:
                window_stop_step += TRANSCRIPTION_CHUNK_STEP_SIZE

        else:
            continue

        # Processing accumulated word lists
        if len(word_lists) >= PROCESSING_MIN_TIMESPAN_DONE:
            max_dupes = find_duplicate_words(word_lists)
            confirmed, potential = get_confirmed_potential_words(max_dupes, word_lists)

            if confirmed is None and potential is None:
                continue

            confirmed_transcribed += join_words(confirmed)
            transcribed = join_words(potential)

            # Update window position based on last confirmed word
            last_confirmed_word = confirmed[-1]
            window_start_step += last_confirmed_word.end * 1
            window_stop_step = math.ceil(window_start_step + TRANSCRIPTION_CHUNK_STEP_SIZE)
            word_lists.clear()

        log.info(f"Confirmed: {confirmed_transcribed} | Potential: {transcribed}")
        time.sleep(PROCESSING_DELAY)

    play_sound(SoundEvent.PROCESSING_END)
    to_clipboard(confirmed_transcribed)
    to_console(confirmed_transcribed)
    log.info("End of consumer thread")


if __name__ == "__main__":
    whisper = WhisperModel(MODEL_NAME, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE, download_root=MODEL_DIR)
    try:
        producer_thread = threading.Thread(target=producer, args=())
        producer_thread.start()

        consumer_thread = threading.Thread(target=consumer, args=())
        consumer_thread.start()

        consumer_thread.join()

        is_recording = False
        producer_thread.join()

    except Exception as e:
        log.error(f"Error in processing: {e}")
    finally:
        gc.collect()


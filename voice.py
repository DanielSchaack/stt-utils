import time
import threading
import queue
import logging
import math

import gc
from typing import Any
from dataclasses import dataclass
from faster_whisper import WhisperModel

import numpy as np
import pyaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
LANGUAGE = "de"
MAX_TIMESPAN = 10
MIN_TIMESPAN_DONE = 5
PROCESSING_DELAY = 0.1
MIN_DUPE_WORD_COUNT = 2
MIN_DUPE_BETWEEN_RECORDS_NEEDED = 2
q = queue.Queue()
is_recording = False


# a Word is (start: np.ndfloat64, end: np.ndfloat64, word: str (usually '<Space>word'), probability: np.ndfloat64)
@dataclass
class Word:
    start: np.float64
    end: np.float64
    probability: np.float64
    word: str


def callback_send_to_queue(in_data, frame_count, time_info, status):
    log.debug(f"Received {len(in_data)} bytes of audio data")
    q.put(in_data)
    return (in_data, pyaudio.paContinue)


def producer():
    global is_recording
    p = pyaudio.PyAudio()
    log.info("Starting audio recording...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=int(CHUNK * 0.5),
                    stream_callback=callback_send_to_queue)

    is_recording = True
    while is_recording:
        time.sleep(PROCESSING_DELAY)

    sounds(False)
    stream.close()
    p.terminate()
    log.info("End of producer thread")
    return


def consumer():
    log.info("Consumer thread started")
    total_data = b''
    window_start_step = 0
    window_stop_step = 1
    available_chunks = 0
    transcribed = ""
    confirmed_transcribed = ""
    potential_transcribed = ""
    word_lists = []
    do_run = True
    while do_run:
        log.info(f"current confirmed transcription: {confirmed_transcribed}")
        log.info(f"current potential transcription: {potential_transcribed}")
        if window_stop_step - window_start_step > MIN_TIMESPAN_DONE and not transcribed.strip():
            do_run = False

        if q.qsize() != 0:
            total_data += q.get()
            q.task_done()

        available_chunks = len(total_data) // CHUNK
        log.info(f"Available chunks: {available_chunks}, current size: {len(total_data)}, current start step: {window_start_step}, current stop step: {window_stop_step}")

        if window_stop_step <= available_chunks:
            np_data = get_window(total_data, window_start_step, window_stop_step, CHUNK)
            transcribed, all_words = transcribe_window(whisper, np_data, confirmed_transcribed)
            log.info(f"Chunk {window_stop_step} transcribed: {transcribed}")

            if len(all_words) > 0:
                word_lists.append(all_words)

            if window_stop_step - window_start_step == MAX_TIMESPAN:
                log.debug(f"Reached max timespan {MAX_TIMESPAN}, evaluating current words to move on")
            else:
                window_stop_step += 1

        if len(word_lists) >= MIN_TIMESPAN_DONE:
            max_dupes = get_max_dupe(word_lists)
            log.debug(word_lists)
            log.info(f"max dupes: {max_dupes}")

            confirmed, potential = get_confirmed_potential_words(max_dupes, word_lists)
            log.debug(f"confirmed words: {confirmed}")
            log.debug(f"potential words: {potential}")

            if confirmed is None and potential is None:
                continue

            confirmed_transcribed += get_printed_words(confirmed)
            potential_transcribed = get_printed_words(potential)
            log.info(f"new confirmed transcription: {confirmed_transcribed}")
            log.info(f"new potential transcription: {potential_transcribed}")

            last_confirmed_word = confirmed[-1]
            log.info(f"last confirmed word: {last_confirmed_word}")
            window_start_step += last_confirmed_word.end
            window_stop_step = math.ceil(window_start_step + 1)
            word_lists.clear()

        time.sleep(PROCESSING_DELAY)
    log.info("End of consumer thread")


def get_confirmed_potential_words(dupe_lists: list[tuple[list[int], list[int]]], word_lists: list[list[Word]]) -> Any:
    index = len(word_lists) - 1
    for dupe_list in reversed(dupe_lists):
        list_amount_dupes, list_dupes_to = dupe_list
        log.debug(f"List of duplicate amounts: {list_amount_dupes}")
        log.debug(f"List mapping duplicates to words: {list_dupes_to}")
        if len(list_amount_dupes) >= MIN_DUPE_BETWEEN_RECORDS_NEEDED:
            min_index, min_value = get_index_dupes(list_amount_dupes, list_dupes_to)
            log.debug(f"Minimum index: {min_index}")
            log.debug(f"Minimum value: {min_value}")
            return word_lists[index][:min_value + 1], word_lists[index][min_value:]
        index -= 1
    return None, None


def get_printed_words(word_list: list[Word]) -> str:
    word_str: str = ""
    for word in word_list:
        word_str += word.word
    return word_str


def get_index_dupes(dupes_list: list[int], index_list: list[int]) -> tuple[int, int]:
    min_dupes: int = min(dupes_list)
    index: int = len(index_list) - 1
    for current_dupe in reversed(dupes_list):
        if current_dupe == min_dupes:
            break
        index -= 1
    return index, min_dupes


def get_window(total_data, window_start_step: float, window_stop_step: int, chunksize: int) -> np.ndarray:
    start_index = math.floor(window_start_step * chunksize)
    end_index = window_stop_step * chunksize
    # difference must be a duplicate of 2
    if (end_index - start_index) % 2 == 1:
        start_index -= 1
    log.debug(f"Start index: {start_index}, End index: {end_index}")
    return (np.frombuffer(total_data[start_index:end_index], np.int16)
            .flatten()
            .astype(np.float32) / 32768.0
            )


# a Word is (start: np.ndfloat64, end: np.ndfloat64, word: str (usually '<Space>word'), probability: np.ndfloat64)
def transcribe_window(model: WhisperModel, np_data: np.ndarray, context: str) -> tuple[str, list]:
    segments, _ = model.transcribe(audio=np_data,
                                   language=LANGUAGE,
                                   initial_prompt=context if context != "" else "",
                                   beam_size=5,
                                   without_timestamps=False,
                                   word_timestamps=True,

                                   vad_filter=True,
                                   vad_parameters=dict(min_silence_duration_ms=1000)
                                   )
    all_words = []
    for segment in segments:
        all_words.extend(segment.words)
    log.debug(all_words)
    transcription = "".join(word.word for word in all_words)
    return transcription, all_words


def get_max_dupe(list_of_lists: list[list[Word]]) -> list[list[int], list[int]]:
    """
    """
    max_dupes = []
    len_lists = len(list_of_lists)
    for i in range(len_lists):
        current_dupe = []
        compared_to = []
        for j in range(i + 1, len_lists):
            if j < len_lists:
                amount_dupe, dupe_words = compare_lists_words(list_of_lists[i], list_of_lists[j])
                if amount_dupe > MIN_DUPE_WORD_COUNT:
                    compared_to.append(j)
                    current_dupe.append(amount_dupe)

        max_dupes.append((current_dupe, compared_to))
    return max_dupes


def compare_lists_words(list_a: list[Word], list_b: list[Word]) -> tuple[int, list[Word]]:
    min_len = min(len(list_a), len(list_b))
    duplicate = 0
    words = []
    for i in range(min_len):
        if list_a[i].word.lower() == list_b[i].word.lower():
            duplicate += 1
            words.append(list_a[i].word)
        else:
            return duplicate, words
    return duplicate, words


def sounds(is_sound: bool):
    pass


if __name__ == "__main__":
    whisper = WhisperModel("small", device="cuda", compute_type="float16", download_root="./models")
    # print(whisper.available_models())
    try:
        producer_thread = threading.Thread(target=producer, args=())
        producer_thread.start()
        consumer_thread = threading.Thread(target=consumer, args=())
        consumer_thread.start()

        consumer_thread.join()
        is_recording = False
        producer_thread.join()
    except Exception as e:
        print(e)
    finally:
        # del whisper
        gc.collect()


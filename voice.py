import time
import threading
import queue
import logging

import gc
from typing import Any
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
RECURRING_WORD_COUNT = 2
q = queue.Queue()
is_recording = False


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
        if window_stop_step - window_start_step > MIN_TIMESPAN_DONE and not transcribed.strip():
            do_run = False

        if q.qsize() != 0:
            total_data += q.get()
            q.task_done()

        available_chunks = len(total_data) // CHUNK
        log.debug(f"Available chunks: {available_chunks}, current size: {len(total_data)}, current stop step: {window_stop_step}")

        if window_stop_step <= available_chunks:
            np_data = get_window(total_data, window_start_step, window_stop_step, CHUNK)
            transcribed, all_words = transcribe_window(whisper, np_data, confirmed_transcribed)
            log.info(f"Chunk {window_stop_step} transcribed: {transcribed}")

            if len(all_words) > 0:
                word_lists.append(all_words)

            if window_stop_step - window_start_step < MAX_TIMESPAN:
                window_stop_step += 1

        if len(word_lists) > 0:
            # step_increment, confirmed_words, potential_words, word_lists_new = compare_words(word_lists)
            max_dupes = get_max_dupe(word_lists)
            log.debug(word_lists)
            log.info(f"max dupes: {max_dupes}")

        time.sleep(PROCESSING_DELAY)
    log.info("End of consumer thread")


def get_window(total_data, window_start_step: int, window_stop_step: int, chunksize: int) -> np.ndarray:
    start_index = window_start_step * chunksize
    end_index = window_stop_step * chunksize
    log.debug(f"Start index: {start_index}, End index: {end_index}")
    return (np.frombuffer(total_data[start_index:end_index], np.int16)
            .flatten()
            .astype(np.float32) / 32768.0
            )


# a Word is (start: np.ndfloat64, end: np.ndfloat64, word: str (usually '<Space>word'), probability: np.ndfloat64)
def transcribe_window(model: WhisperModel, np_data: np.ndarray, context: str) -> tuple[str, list]:
    segments, _ = model.transcribe(audio=np_data,
                                   language=LANGUAGE,
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


# TODO word handling
# a Word is (start: np.ndfloat64, end: np.ndfloat64, word: str (usually '<Space>word'), probability: np.ndfloat64)
def compare_words(word_lists: list[Any]) -> tuple[float, list[Any], list[Any], list[Any]]:
    return 0.0, [], [], word_lists


def get_max_dupe(list_of_lists: list[list[Any]]) -> list[Any]:
    """
    Finds the maximum number of duplicate words between any two lists in a given list of lists.

    Parameters:
        list_of_lists (list[list[Any]]): A list containing sublists to be compared for duplicates.

    Returns:
        list[Any]: A list where each element is a tuple containing the maximum number of duplicate
                  words and the index of the second sublist that had the most duplicates with the current sublist.
                  If there are no duplicates, it returns 0 and -1 as the index.

    Example:
    >>> get_max_dupe([['a', 'b', 'c'], ['d', 'e', 'f'], ['m', 'n', 'o'], ['m', 'n', 'd']])
    [(0, -1), (0, -1), (2, 3), (0, -1)]
    """
    max_dupes = []
    len_lists = len(list_of_lists)
    for i in range(len_lists):
        current_max_dupe = 0
        compared_to = -1
        compared_words = []
        for j in range(i + 1, len_lists):
            if j < len_lists:
                amount_dupe, dupe_words = compare_lists_words(list_of_lists[i], list_of_lists[j])
                compared_to = j if amount_dupe > current_max_dupe else compared_to
                compared_words = dupe_words if amount_dupe > current_max_dupe else compared_words
                current_max_dupe = amount_dupe if amount_dupe > current_max_dupe else current_max_dupe

        max_dupes.append((current_max_dupe, compared_to, compared_words))
    return max_dupes


def compare_lists_words(list_a: list[Any], list_b: list[Any]) -> tuple[int, int, list[str]]:
    min_len = min(len(list_a), len(list_b))
    duplicate = 0
    words = []
    for i in range(min_len):
        if list_a[i].word == list_b[i].word:
            duplicate += 1
            words.append(list_a[i].word)
        else:
            return duplicate, words
    return duplicate, words


def sounds(is_sound: bool):
    pass


# the following is for OG OpenAi Whisper
# wasn't able to get model.transcribe() working
# so instead using whisper functions directly did work ... somehow
# def transcribe_window(model: whisper.Whisper, np_data: np.ndarray, context: str) -> str:
#     audio = whisper.pad_or_trim(np_data)
#     mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
#     options = whisper.DecodingOptions(language="de", beam_size=5, fp16=False)
#     result = whisper.decode(model, mel, options)
#     return result.text


if __name__ == "__main__":
    whisper = WhisperModel("small", device="cpu", compute_type="int8", download_root="./models")
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


from typing import Union, List
import time
import threading
import queue

import gc
from faster_whisper import WhisperModel

import numpy as np
import pyaudio

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
    item = (in_data, frame_count)
    q.put(item=item)
    return (in_data, pyaudio.paContinue)


def producer():
    print("Starting producer thread")
    global is_recording
    is_recording = True
    p = pyaudio.PyAudio()

    print("Start of stream")
    sounds(True)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,  # start recording
                    output=True,  # debug live recording
                    frames_per_buffer=int(CHUNK * 0.5),
                    stream_callback=callback_send_to_queue)

    while is_recording:
        time.sleep(PROCESSING_DELAY)

    sounds(False)
    stream.close()
    p.terminate()
    print("End of producer thread")
    return


def consumer():
    print("Starting consumer thread")
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
            (in_data, _) = q.get()
            total_data += in_data
            q.task_done()

        available_chunks = len(total_data) // CHUNK
        print(f"Available chunks: {available_chunks}, current size: {len(total_data)}, current stop step: {window_stop_step}")

        if window_stop_step <= available_chunks:
            np_data = get_window(total_data, window_start_step, window_stop_step, CHUNK)
            transcribed, all_words = transcribe_window(whisper, np_data, confirmed_transcribed)
            print(f"Chunk {window_stop_step} transcribed: {transcribed}")

            if len(all_words) > 0:
                word_lists.append(all_words)

            if window_stop_step - window_start_step < MAX_TIMESPAN:
                window_stop_step += 1

        if len(word_lists) > 0:
            step_increment, confirmed_words, potential_words, word_lists_new = compare_words(word_lists)

        time.sleep(PROCESSING_DELAY)
    print("End of consumer thread")


def get_window(total_data, window_start_step: int, window_stop_step: int, chunksize: int) -> np.ndarray:
    start_index = window_start_step * chunksize
    end_index = window_stop_step * chunksize
    print(f"Start index: {start_index}, End index: {end_index}")
    return (np.frombuffer(total_data[start_index:end_index], np.int16)
            .flatten()
            .astype(np.float32) / 32768.0
            )


# a Word is (start: np.ndfloat64, end: np.ndfloat64, word: str (usually '<Space>word'), probability: np.ndfloat64)
def transcribe_window(model: WhisperModel, np_data: np.ndarray, context: str) -> Union[str, List]:
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
    print(all_words)
    transcription = "".join(word.word for word in all_words)
    return transcription, all_words


# TODO word handling
def compare_words(word_lists: list) -> Union[float, List, List, List]:
    min_len = min(word_lists, key=len)

    for i in range(min_len):
        pass

    return 0.0, [], [], word_lists


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
        del whisper
        gc.collect()


from .config import AppConfig
import time
import threading
import queue
import logging
import math
import gc
import unicodedata

from faster_whisper import WhisperModel
import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf

from typing import Optional
from dataclasses import dataclass
from enum import Enum

recording_logic_thread = None


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


class Transcriptor():
    def __init__(self, sound_dir: str, config: AppConfig):
        self.sound_dir = sound_dir
        self.config = config
        self.log = logging.getLogger(__name__)
        self.audio_queue = queue.Queue()
        self.is_hotkey_pressed_flag = False
        self.is_recording_flag = False
        self.confirmed = ""
        self.potential = ""
        self.eot = False
        self.eos = False
        self.whisper = None
        self.unload_timer = None

    def update_config(self, config: AppConfig):
        self.config = config

    def callback_send_to_queue(self, indata, frames, time, status):
        self.log.info(f"Received {len(indata)} bytes of audio data")
        self.audio_queue.put(indata.copy())
        return None

    def get_audio_window(self, total_data, window_start_step: float, window_stop_step: float, chunksize: int) -> np.ndarray:
        """Extract a window of audio data and convert to numpy array."""
        start_index = math.floor(window_start_step * chunksize)
        end_index = int(window_stop_step * chunksize)

        # Due to required int16, ensure difference is a multiple of 2
        if (end_index - start_index) % 2 == 1:
            start_index -= 1

        self.log.info(f"Start index: {start_index}, End index: {end_index}")
        return (np.frombuffer(total_data[start_index:end_index], np.int16)
                .flatten()
                .astype(np.float32) / 32768.0
                )

    def transcribe_window(self, np_data: np.ndarray, context: str) -> tuple[str, list[Word]]:
        """Transcribe audio data using the Whisper model."""
        segments, _ = self.whisper.transcribe(audio=np_data,
                                              language=self.config.transcription.language,
                                              initial_prompt=self.config.transcription.nudge_into_punctuation + context if context != "" else self.config.transcription.nudge_into_punctuation,
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

        self.log.debug(f"Described with context '{context} the words: {all_words}")
        transcription = "".join(word.word for word in all_words)
        return transcription, all_words

    def find_duplicate_words(self, list_of_lists: list[list[Word]]) -> list[tuple[list[int], list[int]]]:
        """Find duplicates between different word lists."""
        max_dupes = []
        len_lists = len(list_of_lists)

        for i in range(len_lists):
            current_dupe = []
            compared_to = []

            for j in range(i + 1, len_lists):
                if j < len_lists:
                    amount_dupe, dupe_words = self.compare_word_lists(list_of_lists[i], list_of_lists[j])
                    self.log.debug(f"Amount dupes: {amount_dupe} with duplicate words: {dupe_words}")

                    if amount_dupe >= self.config.processing.min_dupe_word_count:
                        compared_to.append(j)
                        current_dupe.append(amount_dupe)

            max_dupes.append((current_dupe, compared_to))
        self.log.debug(f"Calulated duplicate words: {max_dupes}")
        return max_dupes

    def normalize_string(self, s):
        return unicodedata.normalize('NFC', s.strip()).casefold()

    def compare_word_lists(self, list_a: list[Word], list_b: list[Word]) -> tuple[int, list[str]]:
        """Compare two word lists and count matching words."""
        min_len = min(len(list_a), len(list_b))
        duplicate = 0
        words = []

        for i in range(min_len):
            word_a = self.normalize_string(list_a[i].word)
            word_b = self.normalize_string(list_b[i].word)
            self.log.debug(f"Comparing {i}th word '{word_a}' with '{word_b}'")
            if word_a == word_b:
                duplicate += 1
                words.append(list_a[i].word)
            else:
                return duplicate, words

        return duplicate, words

    def get_index_dupes(self, dupes_list: list[int], index_list: list[int]) -> tuple[int, int]:
        """Find the index of the minimum duplication count."""
        min_dupes: int = min(dupes_list)
        index: int = len(index_list) - 1

        for current_dupe in reversed(dupes_list):
            if current_dupe == min_dupes:
                break
            index -= 1

        return index, min_dupes

    def get_confirmed_potential_words(self, dupe_lists: list[tuple[list[int], list[int]]],
                                      word_lists: list[list[Word]]) -> tuple[Optional[list[Word]], Optional[list[Word]]]:
        """Determine which words are confirmed and which are potentially available."""
        index = len(word_lists) - 1

        for dupe_list in reversed(dupe_lists):
            list_amount_dupes, list_dupes_to = dupe_list
            self.log.debug(f"List of duplicate amounts: {list_amount_dupes}")
            self.log.debug(f"List mapping duplicates to words: {list_dupes_to}")

            if len(list_amount_dupes) >= self.config.processing.min_dupe_between_records_needed:
                min_index, min_value = self.get_index_dupes(list_amount_dupes, list_dupes_to)
                return word_lists[index][:min_value], word_lists[index][min_value:]
            index -= 1
        return None, None

    def join_words(self, word_list: list[Word]) -> str:
        """Join a list of words into a string."""
        return "".join(word.word for word in word_list)

    def to_clipboard(self, text: str):
        if self.config.transcription.to_clipboard and text:
            text_to_clipboard = text.strip()
            self.log.info(f"Adding to clipboard: '{text_to_clipboard}'")
            pyperclip.copy(text_to_clipboard)

    def to_terminal(self, text: str):
        if self.config.transcription.to_terminal and text:
            text_to_terminal = text.strip()
            self.log.info(f"Printing to stdout: '{text_to_terminal}'")
            print(text_to_terminal, flush=True)

    def play_sound_async(self, sound: SoundEvent):
        self.log.debug(f"Playing sound {sound} asynchrounously")
        sound_thread = threading.Thread(target=self.play_sound, args=(sound,))
        sound_thread.start()

    def play_sound(self, sound: SoundEvent):
        self.log.debug(f"Playing sound {sound}")
        if sound == SoundEvent.RECORDING_START and self.config.sound.recording_start_active:
            file_path = self.sound_dir + "/start_recording.wav"
        elif sound == SoundEvent.RECORDING_END and self.config.sound.recording_end_active:
            file_path = self.sound_dir + "/end_recording.wav"
        elif sound == SoundEvent.PROCESSING_END and self.config.sound.processing_end_active:
            file_path = self.sound_dir + "/end_processing.wav"
        else:
            return

        self.log.info(f"Playing sound {file_path}")
        try:
            data, samplerate = sf.read(file_path, dtype='float32', always_2d=True)
            audiolength = math.ceil(len(data) * self.config.sound.relative_length)
            audio = data[:audiolength - 1]
            audio *= self.config.sound.relative_volume
            samplerate *= self.config.sound.relative_speed

            stream = sd.OutputStream(samplerate=samplerate, channels=audio.shape[1])
            with stream:
                stream.write(audio)
        except Exception as e:
            self.log.error(e)

    def unload_whisper(self):
        self.log.info("Unloading Whisper model due to inactivity...")
        self.whisper = None  # Modell freigeben
        gc.collect()  # Garbage Collector manuell ausführen

    def producer(self):
        """Record audio and add it to the queue."""
        amount_wait_intervals = 0

        self.play_sound_async(SoundEvent.RECORDING_START)
        self.log.info("Starting audio recording...")

        stream = sd.InputStream(
            samplerate=self.config.recording.rate,
            blocksize=self.config.recording.chunk_size,
            channels=self.config.recording.channels,
            dtype='int16',
            callback=self.callback_send_to_queue,
            clip_off=True
        )

        self.is_recording_flag = True
        with stream:
            self.log.info("Start of recording stream")
            while self.is_hotkey_pressed_flag:
                amount_wait_intervals += 1
                time.sleep(self.config.processing.delay)
            self.log.info("End of recording stream")

        current_wait_interval = amount_wait_intervals % self.config.processing.delays_per_second
        intervals_to_wait = self.config.processing.delays_per_second - current_wait_interval + 1
        self.log.debug(f"Sleeping {intervals_to_wait} until next full chunk before setting end_of_processing flag")
        time.sleep(self.config.processing.delay * intervals_to_wait)
        self.is_recording_flag = False

        self.play_sound_async(SoundEvent.RECORDING_END)
        self.log.info("End of producer thread")

    def consumer(self):
        """Process audio data from the queue and transcribe it."""
        self.log.info("Consumer thread started")

        # Wait until recordings are available
        if not self.is_recording_flag:
            time.sleep(self.config.processing.delay)
        self.log.info("Recording is active, consumer thread logic starting now")
        self.eot = False

        total_data = np.array([], dtype='int16')
        window_start_step = 0
        window_stop_step = self.config.transcription.chunk_step_size
        available_chunks = 0
        transcribed = ""
        confirmed_transcribed = ""
        word_lists = []
        do_run = True
        last_run = False

        while do_run:
            if self.audio_queue.qsize() != 0:
                self.log.info("Appending to total data")
                total_data = np.append(total_data, self.audio_queue.get())
                self.audio_queue.task_done()
            else:
                self.log.debug("Not appending data")
                if not self.is_recording_flag and window_stop_step > available_chunks:
                    do_run = False
                    last_run = True
                    self.log.info("Stopping processing")

            available_chunks = len(total_data) // self.config.recording.chunk_size
            self.log.debug(f"Available chunks: {available_chunks}, current size: {len(total_data)}, current start step: {window_start_step}, current stop step: {window_stop_step}")

            # Process available audio data or sleep until one is available
            if window_stop_step <= available_chunks:
                if last_run:
                    window_stop_step = len(total_data) / self.config.recording.chunk_size
                    self.log.info(f"Last run from {window_start_step} till {window_stop_step}")
                    np_data = self.get_audio_window(total_data, window_start_step, window_stop_step, self.config.recording.chunk_size)
                else:
                    self.log.info(f"Last run from {window_start_step} till {window_stop_step}")
                    np_data = self.get_audio_window(total_data, window_start_step, window_stop_step, self.config.recording.chunk_size)
                transcribed, all_words = self.transcribe_window(np_data, confirmed_transcribed)
                self.log.info(f"From {window_start_step:.2f} seconds to {window_stop_step:.2f} seconds transcribed to: {transcribed}")

                if len(all_words) > 0:
                    word_lists.append(all_words)

                if window_stop_step - math.ceil(window_start_step) >= self.config.transcription.max_timespan:
                    self.log.info(f"Reached max timespan {self.config.transcription.max_timespan}, evaluating current words to move on")
                    confirmed_transcribed += transcribed
                    transcribed = ""
                    last_confirmed_word = all_words[-1]
                    window_start_step += last_confirmed_word.end
                    window_stop_step = math.ceil(window_start_step + self.config.transcription.chunk_step_size)
                    word_lists.clear()
                    continue
                else:
                    window_stop_step += self.config.transcription.chunk_step_size

            else:
                time.sleep(self.config.processing.delay)
                continue

            # Processing accumulated word lists
            if len(word_lists) >= self.config.processing.min_timespan_done:
                max_dupes = self.find_duplicate_words(word_lists)
                confirmed, potential = self.get_confirmed_potential_words(max_dupes, word_lists)

                if confirmed is None and potential is None:
                    continue

                confirmed_transcribed += self.join_words(confirmed)
                transcribed = self.join_words(potential)

                # Update window position based on last confirmed word
                last_confirmed_word = confirmed[-1]
                window_start_step += last_confirmed_word.end * 1
                window_stop_step = math.ceil(window_start_step + self.config.transcription.chunk_step_size)
                word_lists.clear()
                if self.config.transcription.terminal_share_progress:
                    self.to_terminal(confirmed_transcribed + self.config.transcription.separation_confirmed_potential + transcribed)

            self.confirmed = confirmed_transcribed
            self.potential = transcribed
            self.log.info(f"Confirmed: {confirmed_transcribed} | Potential: {transcribed}")

        self.play_sound_async(SoundEvent.PROCESSING_END)
        self.to_clipboard(confirmed_transcribed + self.config.transcription.separation_confirmed_potential + transcribed)
        self.to_terminal(confirmed_transcribed + self.config.transcription.separation_confirmed_potential + transcribed)
        self.confirmed = confirmed_transcribed
        self.potential = transcribed
        self.eot = True
        if self.config.transcription.terminal_share_progress:
            self.to_terminal(self.config.transcription.terminal_eot)
        self.log.info("End of consumer thread")

    def main_logic(self):
        self.log.info("Starting transcription logic")
        if self.whisper is None:
            self.whisper = WhisperModel(self.config.model.name, device=self.config.model.device, compute_type=self.config.model.compute_type, download_root=self.config.model.dir)
            self.log.info("Whisper model loaded")

        if self.unload_timer:
            self.unload_timer.cancel()

        try:
            producer_thread = threading.Thread(target=self.producer, args=())
            producer_thread.start()

            consumer_thread = threading.Thread(target=self.consumer, args=())
            consumer_thread.start()

            consumer_thread.join()
        except Exception as e:
            self.log.error(f"Error in processing: {e}")
        finally:
            self.unload_timer = threading.Timer(self.config.processing.keep_alive, self.unload_whisper)
            self.unload_timer.start()

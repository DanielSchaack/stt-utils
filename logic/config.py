import argparse
import yaml
from dataclasses import dataclass


@dataclass
class TranscriptionConfig:
    language: str
    nudge_into_punctuation: str
    chunk_step_size: int
    max_timespan: int
    to_clipboard: bool
    to_terminal: bool
    terminal_share_progress: bool
    separation_confirmed_potential: str
    terminal_share_eos: bool
    terminal_eos: str
    terminal_eot: str


@dataclass
class RecordingConfig:
    channels: int
    rate: int
    chunk_size: int


@dataclass
class ProcessingConfig:
    delay: float
    delays_per_second: float
    stop_timespan_done: int
    min_timespan_done: int
    min_dupe_word_count: int
    min_dupe_between_records_needed: int


@dataclass
class SoundConfig:
    recording_start_active: bool
    recording_end_active: bool
    processing_end_active: bool
    relative_length: float
    relative_volume: float
    relative_speed: float


@dataclass
class ModelConfig:
    name: str
    device: str
    compute_type: str
    dir: str


@dataclass
class AppConfig:
    transcription: TranscriptionConfig
    recording: RecordingConfig
    processing: ProcessingConfig
    sound: SoundConfig
    model: ModelConfig


def load_config_from_yaml(filename: str) -> AppConfig:
    with open(filename, 'r') as file:
        config_data = yaml.safe_load(file)

    transcription_config = TranscriptionConfig(
        **config_data['transcription']
    )
    recording_config = RecordingConfig(
        **config_data['recording']
    )
    processing_config = ProcessingConfig(
        **config_data['processing']
    )
    sound_config = SoundConfig(
        **config_data['sound']
    )
    model_config = ModelConfig(
        **config_data['model']
    )

    app_config = AppConfig(
        transcription=transcription_config,
        recording=recording_config,
        processing=processing_config,
        sound=sound_config,
        model=model_config
    )

    return app_config


def load_default_config() -> AppConfig:
    """Load default configuration values."""
    return AppConfig(
        transcription=TranscriptionConfig(
            language="de",
            nudge_into_punctuation="Transkription mit Fokus auf Groß- und Kleinschreibung sowie Satzzeichen. ",
            chunk_step_size=1,
            max_timespan=10,
            to_clipboard=True,
            to_terminal=True,
            terminal_share_progress=False,
            separation_confirmed_potential="",
            terminal_share_eos=False,
            terminal_eos=">><<",
            terminal_eot="<><>"

        ),

        recording=RecordingConfig(
            channels=1,
            rate=16000,
            chunk_size=16000
        ),

        processing=ProcessingConfig(
            delay=0.1,
            delays_per_second=1.0 / 0.1,
            stop_timespan_done=5,
            min_timespan_done=2,
            min_dupe_word_count=2,
            min_dupe_between_records_needed=2
        ),

        sound=SoundConfig(
            recording_start_active=True,
            recording_end_active=True,
            processing_end_active=True,
            relative_length=1.0,
            relative_volume=1.0,
            relative_speed=2.0
        ),

        model=ModelConfig(
            name="turbo",
            device="cuda",
            compute_type="float16",
            dir="./models"
        )
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure transcription and model parameters.")
    parser.add_argument("-L", "--transcription_language", type=str, default="de", help="Language of transcription (default: de)")
    parser.add_argument("-c", "--transcription_to_clipboard", action="store_false", help="Copy transcriptions to clipboard")
    parser.add_argument("-t", "--transcription_to_terminal", action="store_true", help="Print transcriptions to terminal")
    parser.add_argument("-p", "--transcription_to_terminal_share_progress", action="store_true", help="Print progressional transcriptions to terminal")
    parser.add_argument("-eot", "--transcription_to_terminal_eot", default="<><>", help="Terminator of transcriptions in case of progressive prints, end of transcription")
    parser.add_argument("-e", "--transcription_to_terminal_share_eos", action="store_true", help="Print end of service terminator to terminal")
    parser.add_argument("-eos", "--transcription_to_terminal_eos", default=">><<", help="Terminator for following components to be alerted of end of service")
    parser.add_argument("-S", "--transcription_separation_confirmed_potential", type=str, default="", help="Separate confirmed and potential transcriptions with provided string")

    parser.add_argument("-M", "--model_name", type=str, default="turbo", help="Name of the model (default: turbo)")
    parser.add_argument("-D", "--model_device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device for model execution (cuda or cpu, default: cuda)")
    parser.add_argument("-C", "--model_compute_type", type=str, choices=["float16", "int8"], default="float16", help="Computation type of the model (float16 or int8, default: float16)")
    return parser.parse_args()


class ConfigManager:
    def __init__(self):
        self.config = load_default_config()

    def load_from_file(self, filename: str) -> bool:
        file_config = load_config_from_yaml(filename)
        if file_config:
            self.config = file_config
            return self.config
        return self.config

    def parse_args(self):
        args = parse_arguments()
        self.config.transcription.language = args.transcription_language
        self.config.transcription.to_clipboard = args.transcription_to_clipboard
        self.config.transcription.to_terminal = args.transcription_to_terminal
        self.config.transcription.terminal_share_progress = args.transcription_to_terminal_share_progress
        self.config.transcription.terminal_eot = args.transcription_to_terminal_eot
        self.config.transcription.terminal_share_eos = args.transcription_to_terminal_share_eos
        self.config.transcription.terminal_eos = args.transcription_to_terminal_eos
        self.config.transcription.separation_confirmed_potential = args.transcription_separation_confirmed_potential
        self.config.model.name = args.model_name
        self.config.model.device = args.model_device
        self.config.model.compute_type = args.model_compute_type

    def get_config(self) -> AppConfig:
        return self.config

from collections import deque
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import subprocess
from threading import Event

import av
import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel
import numpy as np
import psutil

RATE = 16_000
WINDOW = 300 * RATE
OVERLAP = 2 * RATE
_DLL_HANDLES = []
_DLL_PATHS = set()

MODELS = {
    "tiny": "Fastest",
    "base": "Fast",
    "small": "Balanced",
    "medium": "Higher accuracy",
    "turbo": "Fast large model",
    "large-v3": "Full large model",
    "tiny.en": "Fastest English",
    "base.en": "Fast English",
    "small.en": "Balanced English",
    "medium.en": "Higher accuracy English",
    "distil-large-v3.5": "Fast large English model",
}

LANGUAGES = {
    "Auto detect": None, "Arabic": "ar", "Bengali": "bn", "Chinese": "zh",
    "Dutch": "nl", "English": "en", "French": "fr", "German": "de",
    "Gujarati": "gu", "Hindi": "hi", "Indonesian": "id", "Italian": "it",
    "Japanese": "ja", "Kannada": "kn", "Korean": "ko", "Malayalam": "ml",
    "Marathi": "mr", "Nepali": "ne", "Portuguese": "pt", "Punjabi": "pa",
    "Russian": "ru", "Spanish": "es", "Tamil": "ta", "Telugu": "te",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur",
    "Vietnamese": "vi",
}
LANGUAGES.update({
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Armenian": "hy",
    "Assamese": "as", "Azerbaijani": "az", "Bashkir": "ba", "Basque": "eu",
    "Belarusian": "be", "Bosnian": "bs", "Breton": "br", "Bulgarian": "bg",
    "Burmese": "my", "Cantonese": "yue", "Catalan": "ca", "Croatian": "hr", "Czech": "cs",
    "Danish": "da", "Estonian": "et", "Faroese": "fo", "Finnish": "fi",
    "Galician": "gl", "Georgian": "ka", "Greek": "el", "Haitian Creole": "ht",
    "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he", "Hungarian": "hu",
    "Icelandic": "is", "Javanese": "jw", "Kazakh": "kk", "Khmer": "km",
    "Lao": "lo", "Latin": "la", "Latvian": "lv", "Lingala": "ln", "Lithuanian": "lt",
    "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms",
    "Maltese": "mt", "Maori": "mi", "Mongolian": "mn", "Norwegian": "no",
    "Norwegian Nynorsk": "nn", "Occitan": "oc", "Pashto": "ps",
    "Persian": "fa", "Polish": "pl", "Romanian": "ro", "Sanskrit": "sa",
    "Serbian": "sr", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Sundanese": "su",
    "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg",
    "Tatar": "tt", "Tibetan": "bo", "Turkmen": "tk", "Uzbek": "uz",
    "Welsh": "cy", "Yiddish": "yi", "Yoruba": "yo",
})
ENGLISH_ONLY = {name for name in MODELS if name.endswith(".en") or name.startswith("distil-")}
CANTONESE_MODELS = {"large-v3", "turbo"}


@dataclass(frozen=True)
class Hardware:
    gpu: str = ""
    index: int = 0
    free_mib: int = 0
    compute: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    model: str
    device: str
    compute: str
    index: int = 0
    batch: int = 1


class Cancelled(Exception):
    pass


def detect_hardware():
    try:
        site_packages = Path(ctranslate2.__file__).resolve().parent.parent
        libraries = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")
        bundled = [site_packages / "nvidia" / library / "bin" for library in libraries]
        required = ("cublas64_12.dll", "cudnn64_9.dll")
        external = [Path(entry.strip('"')) for entry in os.environ.get("PATH", "").split(os.pathsep)]
        for directory in bundled + [
            path for path in external if path.is_absolute() and any((path / dll).is_file() for dll in required)
        ]:
            path = str(directory)
            if directory.is_dir() and path not in _DLL_PATHS:
                _DLL_HANDLES.append(os.add_dll_directory(path))
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                _DLL_PATHS.add(path)
        for dll in required:
            ctypes.WinDLL(dll)
        count = ctranslate2.get_cuda_device_count()
        if not count:
            return Hardware(reason="No CUDA device found")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=4, creationflags=flags,
            ).splitlines()
            devices = {
                uuid: (name, int(memory))
                for row in rows
                for uuid, name, memory in [[part.strip() for part in row.split(",")]]
            }
            # Select ordinal zero to respect CUDA_VISIBLE_DEVICES. Use the lowest
            # reported free memory when the ordinal cannot be mapped confidently.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            if visible.isdecimal():
                visible = list(devices)[int(visible)]
            gpu, free = devices.get(visible) or min(devices.values(), key=lambda item: item[1])
            if len(devices) > 1 and not visible:
                gpu = "NVIDIA CUDA GPU"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            gpu, free = "NVIDIA CUDA GPU", 0
        types = ctranslate2.get_supported_compute_types("cuda", 0)
        compute = next((kind for kind in ("int8_float16", "int8_float32", "float16", "float32")
                        if kind in types), "")
        return Hardware(gpu, 0, free, compute, "" if compute else "No supported CUDA precision")
    except (OSError, RuntimeError) as error:
        return Hardware(reason=str(error))


def choose_plan(hardware, model, language, device):
    use_gpu = device != "CPU" and bool(hardware.compute)
    if device == "NVIDIA GPU" and not use_gpu:
        raise RuntimeError(f"CUDA unavailable. {hardware.reason}")
    if model == "Auto":
        if use_gpu:
            # Conservative memory estimates for quantized inference.
            free = hardware.free_mib
            if hardware.compute in {"float16", "float32"}:
                free //= 2
            model = ("large-v3" if free >= 8500 else
                     "distil-large-v3.5" if free >= 4500 and language == "en" else
                     "turbo" if free >= 4500 else "small" if free >= 2000 else
                     "base" if free >= 1000 else "tiny")
        else:
            memory = psutil.virtual_memory().available
            cores = psutil.cpu_count(logical=False) or 1
            model = "small" if memory >= 6 * 1024**3 and cores >= 8 else "base"
            if memory < 2 * 1024**3:
                model = "tiny"
        if language == "en" and f"{model}.en" in MODELS:
            model += ".en"
        if language == "yue" and model not in CANTONESE_MODELS:
            model = "turbo"
    if model in ENGLISH_ONLY and language != "en":
        raise ValueError("Select English before using an English-only model.")
    if language == "yue" and model not in CANTONESE_MODELS:
        raise ValueError("Cantonese requires Turbo or Large v3.")
    if use_gpu:
        batch = 4 if hardware.free_mib >= 6000 else 2 if hardware.free_mib >= 2800 else 1
        return Plan(model, "cuda", hardware.compute, hardware.index, batch)
    return Plan(model, "cpu", "int8")


def audio_windows(container, stream, stop):
    """Yield (offset, owned start, owned end, samples) on a continuous audio timeline."""
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
    pending = deque()
    samples = 0
    offset = 0
    left = 0

    def frames():
        for frame in container.decode(stream):
            if stop.is_set():
                raise Cancelled
            yield from resampler.resample(frame)
        yield from resampler.resample(None)

    for frame in frames():
        part = frame.to_ndarray().reshape(-1)
        pending.append(part)
        samples += part.size
        if samples >= WINDOW:
            contiguous = np.concatenate(pending)
            yield offset, left, WINDOW - OVERLAP, contiguous[:WINDOW]
            advance = WINDOW - 2 * OVERLAP
            pending = deque([contiguous[advance:].copy()])
            samples -= advance
            offset += advance
            left = OVERLAP
    if samples > left:
        yield offset, left, samples, np.concatenate(pending)


def timecode(seconds, separator=","):
    ms = max(0, round(seconds * 1000))
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{ms:03}"


def transcribe(source, target, plan, language, stop: Event, report, detect_speakers=False):
    lines = 0
    chosen_language = language
    txt_path, srt_path = target / "transcript.txt", target / "subtitles.srt"
    with av.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError("The file has no audio track.")
        stream = container.streams.audio[0]
        media_start = (container.start_time or 0) / av.time_base
        audio_start = max(0, float((stream.start_time or 0) * stream.time_base) - media_start)
        duration = (
            container.duration / av.time_base if container.duration else
            audio_start + float(stream.duration * stream.time_base) if stream.duration else 0
        )
        model = WhisperModel(plan.model, device=plan.device,
                             device_index=plan.index, compute_type=plan.compute,
                             cpu_threads=min(psutil.cpu_count(logical=False) or 1, 8))
        pipeline = BatchedInferencePipeline(model) if plan.batch > 1 else None

        def check_cancelled():
            if stop.is_set():
                raise Cancelled

        speaker_detector = None
        if detect_speakers:
            from speakers import SpeakerDetector, labeled_lines

            check_cancelled()
            speaker_detector = SpeakerDetector(check_cancelled)
        with txt_path.open("w", encoding="utf-8") as txt, srt_path.open("w", encoding="utf-8") as srt:
            for offset, left, right, samples in audio_windows(container, stream, stop):
                check_cancelled()
                if np.max(np.abs(samples)) < 0.0001:
                    report(audio_start + (offset + right) / RATE, duration, chosen_language, None)
                    continue
                tracks = []
                if speaker_detector:
                    report(audio_start + (offset + left) / RATE, duration, chosen_language, None)
                    tracks = speaker_detector.tracks(samples)
                    check_cancelled()
                options = dict(language=chosen_language, beam_size=3, vad_filter=True,
                               word_timestamps=bool(tracks))
                if pipeline and chosen_language:
                    segments, info = pipeline.transcribe(samples, batch_size=plan.batch, **options)
                else:
                    segments, info = model.transcribe(samples, **options)
                for segment in segments:
                    check_cancelled()
                    midpoint = (segment.start + segment.end) / 2 * RATE
                    line = segment.text.strip()
                    if not line or not left <= midpoint < right:
                        continue
                    chosen_language = chosen_language or info.language
                    parts = labeled_lines(segment, tracks) if tracks else [(segment.start, segment.end, line)]
                    for start, end, text in parts:
                        start += audio_start + offset / RATE
                        end += audio_start + offset / RATE
                        lines += 1
                        txt.write(text + "\n")
                        srt.write(f"{lines}\n{timecode(start)} --> {timecode(end)}\n{text}\n\n")
                        report(start, duration, chosen_language, text)
                txt.flush()
                srt.flush()
                report(audio_start + (offset + right) / RATE, duration, chosen_language, None)
    if stop.is_set():
        raise Cancelled
    return lines

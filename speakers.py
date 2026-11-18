"""Optional offline speaker labels for overlapping audio windows."""

from collections import defaultdict
from hashlib import sha256
import os
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import sherpa_onnx as sherpa


_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
_SEGMENTATION = (
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
)
_EMBEDDING = (
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b",
)


def _download(asset, destination, check_cancelled):
    location, expected_hash = asset
    digest = sha256()
    with urlopen(_RELEASES + location, timeout=30) as source, destination.open("wb") as target:
        while block := source.read(1024 * 1024):
            check_cancelled()
            digest.update(block)
            target.write(block)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError("The speaker model download failed its checksum.")


def _models(check_cancelled):
    cache = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Vid2text" / "speaker-models"
    cache.mkdir(parents=True, exist_ok=True)
    segmentation, embedding = cache / "segmentation-3.0.onnx", cache / "eres2net-base.onnx"
    if segmentation.is_file() and embedding.is_file():
        return segmentation, embedding
    with TemporaryDirectory(dir=cache) as scratch:
        scratch = Path(scratch)
        if not segmentation.is_file():
            archive = scratch / "model.tar.bz2"
            _download(_SEGMENTATION, archive, check_cancelled)
            unpacked = scratch / segmentation.name
            with tarfile.open(archive, "r:bz2") as package:
                source = package.extractfile("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
                if source is None:
                    raise RuntimeError("The speaker segmentation archive is missing its model.")
                with source, unpacked.open("wb") as target:
                    shutil.copyfileobj(source, target)
            unpacked.replace(segmentation)
        if not embedding.is_file():
            unpacked = scratch / embedding.name
            _download(_EMBEDDING, unpacked, check_cancelled)
            unpacked.replace(embedding)
    return segmentation, embedding


class SpeakerDetector:
    def __init__(self, check_cancelled):
        segmentation, embedding = _models(check_cancelled)
        self.embedding = sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=2)
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                num_threads=2,
            ),
            embedding=self.embedding,
            clustering=sherpa.FastClusteringConfig(num_clusters=-1, threshold=0.7),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        self.diarizer = sherpa.OfflineSpeakerDiarization(config)
        self.extractor = sherpa.SpeakerEmbeddingExtractor(self.embedding)
        self.prototypes = []

    def tracks(self, samples):
        regions = self.diarizer.process(samples).sort_by_start_time()
        by_speaker = defaultdict(list)
        rate = self.diarizer.sample_rate
        for region in regions:
            by_speaker[region.speaker].append(region)

        available = set(range(len(self.prototypes)))
        labels = {}
        for local, turns in by_speaker.items():
            longest = sorted(turns, key=lambda turn: turn.end - turn.start, reverse=True)
            clips = [
                samples[round(turn.start * rate):round(min(turn.end, turn.start + 5) * rate)]
                for turn in longest[:3]
            ]
            stream = self.extractor.create_stream()
            stream.accept_waveform(rate, np.concatenate(clips))
            stream.input_finished()
            vector = None
            if self.extractor.is_ready(stream):
                vector = np.asarray(self.extractor.compute(stream))
                norm = np.linalg.norm(vector)
                vector = vector / norm if norm else None
            matches = {
                index: float(np.dot(self.prototypes[index], vector))
                for index in available
                if vector is not None and self.prototypes[index] is not None
            }
            best = max(matches, key=matches.get) if matches else None
            if best is None or matches[best] < 0.65:
                best = len(self.prototypes)
                self.prototypes.append(vector)
            else:
                available.remove(best)
                updated = self.prototypes[best] + vector
                self.prototypes[best] = updated / np.linalg.norm(updated)
            labels[local] = best + 1
        return [(region.start, region.end, labels[region.speaker]) for region in regions]


def labeled_lines(segment, tracks):
    """Split a transcript segment when word timestamps indicate a speaker change."""
    def speaker_for(start, end):
        best_overlap, speaker = 0, None
        for left, right, candidate in tracks:
            if left >= end:
                break
            overlap = min(right, end) - max(left, start)
            if overlap > best_overlap:
                best_overlap, speaker = overlap, candidate
        return speaker

    def label(speaker, text):
        return f"Speaker {speaker}: {text.strip()}" if speaker else text.strip()

    if not segment.words or "".join(word.word for word in segment.words).strip() != segment.text.strip():
        yield segment.start, segment.end, label(speaker_for(segment.start, segment.end), segment.text)
        return
    start, end, speaker, words = 0, 0, None, []
    for word in segment.words:
        current = speaker_for(word.start, word.end) or speaker
        if words and current != speaker:
            yield start, end, label(speaker, "".join(words))
            words = []
        if not words:
            start, speaker = word.start, current
        words.append(word.word)
        end = word.end
    if words:
        yield start, end, label(speaker, "".join(words))

<div align="center">

# Vid2text

**Turn recordings into searchable transcripts and subtitles on your own PC.**

![Windows x64](https://img.shields.io/badge/Windows-x64-0078D4)
![NVIDIA CUDA](https://img.shields.io/badge/Acceleration-NVIDIA%20CUDA-76B900)
![Local processing](https://img.shields.io/badge/Processing-Local-6256C9)
![TXT and SRT](https://img.shields.io/badge/Export-TXT%20%2B%20SRT-444444)

[Download for Windows](../../releases/latest) · [Quick start](#quick-start) · [Model guide](#model-guide) · [Build and release](#build-and-release)

</div>

## Quick start

1. Download the **Windows x64 ZIP** from [Releases](../../releases), extract the entire archive, and open **Vid2text.exe**. The executable and its `_internal` folder belong together.
2. Choose an audio or video file. Common formats include MP3, WAV, M4A, FLAC, MP4, MOV, MKV, and WebM. Vid2text uses the first audio track.
3. Leave **Language**, **Model**, and **Device** on their automatic settings, or select a language such as **Hindi (हिन्दी)** or **Bengali (বাংলা)**. Enable **Detect speakers** if you want numbered speaker labels.
4. Click **Transcribe**. Open the output folder for the full `transcript.txt` and `subtitles.srt`.

> The first use of a transcription model downloads its weights. The first use of speaker detection downloads two additional models. Inference and your files stay on your computer.

### Long recordings and speaker labels

Long recordings are decoded in overlapping five-minute windows. The preview shows recent lines while Vid2text writes the full export to disk. Cancellation takes effect between operations and retains any saved output.

**Detect speakers** adds labels such as **Speaker 1** and **Speaker 2** to TXT and SRT. It runs on CPU alongside the transcription workflow and adds processing time. Speaker labels are estimates. Overlapping voices and brief turns may be split or merged, and a returning speaker can receive a new number.

**Auto detect** selects one language per recording. Select the spoken language manually when you know it.

## Model guide

| Model | When to choose it |
| --- | --- |
| **Tiny / Base** | Quick drafts or a low-memory machine |
| **Small** | Everyday transcription on modest hardware |
| **Medium** | More compute for difficult recordings |
| **Turbo** | Fast multilingual large-model inference |
| **Distil Large** | Fast large-model inference for English |
| **Large v3** | Full multilingual model when speed is less important |

English unlocks the English-only variants. Cantonese uses Turbo or Large v3.

### NVIDIA acceleration

The release ZIP includes the CPU runtime. To enable GPU acceleration in the ZIP build, install NVIDIA's CUDA 12 cuBLAS and cuDNN 9 for CUDA 12 and add their DLL directories to your Windows `PATH`. Restart Vid2text after changing `PATH`. The source launcher installs matching runtime packages in its own environment. [Faster Whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)

**Auto** uses CUDA when available, considers free VRAM when selecting a model, and retries on CPU if CUDA inference fails. Select **NVIDIA GPU** to require GPU inference.

## Build and release

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Double-click [Start-Vid2text.cmd](Start-Vid2text.cmd) to run from source with GPU support, or run `uv run python app.py` for CPU use.

Build the Windows executable from source:

```powershell
uv sync --locked --group build
uv run --no-sync pyinstaller --noconfirm Vid2text.spec
```

Open `dist/Vid2text/Vid2text.exe`. Maintainers can publish from a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The [Windows release workflow](.github/workflows/release.yml) builds and smoke-tests the executable, then publishes the ZIP and its SHA-256 checksum. It uses a standard GitHub-hosted Windows runner, which is free for public repositories. [GitHub runner pricing](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories)

Built with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper), [PyAV](https://github.com/PyAV-Org/PyAV), and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

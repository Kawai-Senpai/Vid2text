import os
from pathlib import Path
from queue import Empty, Queue
import sys
import tempfile
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from engine import (
    CANTONESE_MODELS, ENGLISH_ONLY, LANGUAGES, MODELS, Cancelled,
    choose_plan, detect_hardware, timecode, transcribe,
)


class Transcriber:
    def __init__(self, root):
        self.root = root
        root.title("Vid2text")
        root.geometry("800x620")
        root.minsize(560, 475)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        self.language = tk.StringVar(value="Auto detect")
        self.model = tk.StringVar(value="Auto")
        self.device = tk.StringVar(value="Auto")
        self.detect_speakers = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose an audio or video file.")
        self.details = tk.StringVar()
        self.events = Queue()
        self.stop = Event()
        self.worker = None
        self.output = None

        panel = ttk.Frame(root, padding=20)
        panel.grid(sticky="nsew")
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(8, weight=1)
        paths = [
            ("Media file", self.source, self.pick_file),
            ("Save in", self.destination, self.pick_directory),
        ]
        self.settings = []
        for row, (label, variable, callback) in enumerate(paths):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(panel, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", padx=8)
            button = ttk.Button(panel, text="Browse...", command=callback)
            button.grid(row=row, column=2, sticky="ew")
            self.settings.extend([entry, button])

        for row, label, variable, values in [
            (2, "Language", self.language, tuple(sorted(LANGUAGES))),
            (3, "Model", self.model, ()),
            (4, "Device", self.device, ("Auto", "NVIDIA GPU", "CPU")),
        ]:
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
            field = ttk.Combobox(panel, textvariable=variable, values=values, state="readonly")
            field.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(8, 0))
            self.settings.append(field)
            if label == "Model":
                self.models = field
        self.refresh_models()
        self.language.trace_add("write", self.refresh_models)

        actions = ttk.Frame(panel)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 10))
        speaker_checkbox = ttk.Checkbutton(actions, text="Detect speakers", variable=self.detect_speakers)
        speaker_checkbox.pack(side="left", padx=(0, 12))
        self.settings.append(speaker_checkbox)
        self.start_button = ttk.Button(actions, text="Transcribe", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.output_button = ttk.Button(actions, text="Open output folder", command=self.open_output, state="disabled")
        self.output_button.pack(side="right")
        self.bar = ttk.Progressbar(panel, maximum=100)
        self.bar.grid(row=6, column=0, columnspan=3, sticky="ew")
        ttk.Label(panel, textvariable=self.status).grid(row=7, column=0, columnspan=3, sticky="w", pady=6)
        self.preview = ScrolledText(panel, wrap="word", state="disabled", height=10)
        self.preview.grid(row=8, column=0, columnspan=3, sticky="nsew")
        ttk.Label(panel, textvariable=self.details, wraplength=520).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        root.after(100, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def refresh_models(self, *_):
        language = LANGUAGES[self.language.get()]
        values = ["Auto"] + [
            f"{model} - {description}" for model, description in MODELS.items()
            if (language == "en" or model not in ENGLISH_ONLY)
            and (language != "yue" or model in CANTONESE_MODELS)
        ]
        self.models.configure(values=values)
        if self.model.get() not in values:
            self.model.set("Auto")

    def pick_file(self):
        selected = filedialog.askopenfilename(filetypes=[
            ("Audio and video", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpeg *.ts"),
            ("All files", "*.*"),
        ])
        if selected:
            self.source.set(selected)
            if not self.destination.get():
                self.destination.set(str(Path(selected).parent))

    def pick_directory(self):
        selected = filedialog.askdirectory()
        if selected:
            self.destination.set(selected)

    def open_output(self):
        try:
            os.startfile(self.output)
        except OSError as error:
            messagebox.showerror("Could not open output", str(error), parent=self.root)

    def start(self):
        if self.worker is not None:
            return
        source = Path(self.source.get().strip()).expanduser()
        destination = Path(self.destination.get().strip()).expanduser() if self.destination.get().strip() else source.parent
        if not source.is_file() or not destination.is_dir():
            messagebox.showerror("Check file and folder", "Choose a valid media file and an existing output folder.", parent=self.root)
            return
        self.stop.clear()
        self.output = None
        self.output_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        for field in self.settings:
            field.configure(state="disabled")
        self.clear_preview()
        self.details.set("")
        self.status.set("Checking available hardware...")
        self.bar.configure(mode="indeterminate", value=0)
        self.bar.start()
        self.worker = Thread(target=self.run, daemon=True, args=(
            source, destination, self.model.get().split(" - ", 1)[0],
            LANGUAGES[self.language.get()], self.device.get(), self.detect_speakers.get(),
        ))
        self.worker.start()

    def run(self, source, destination, model, language, device, detect_speakers):
        folder = None
        try:
            hardware = detect_hardware() if device != "CPU" else None
            plan = choose_plan(hardware, model, language, device)
            folder = Path(tempfile.mkdtemp(prefix=f"{source.stem[:60]}-transcript-", dir=destination))

            def execute(selected, note=""):
                if self.stop.is_set():
                    raise Cancelled
                label = hardware.gpu if selected.device == "cuda" else "CPU"
                self.events.put(("plan", f"{selected.model} | {label} | {selected.compute}{note}"))
                return transcribe(source, folder, selected, language, self.stop,
                                  lambda *update: self.events.put(("progress", update)),
                                  detect_speakers=detect_speakers)

            try:
                note = " | CUDA unavailable" if device == "Auto" and plan.device == "cpu" else ""
                lines = execute(plan, note)
            except RuntimeError as error:
                message = str(error).lower()
                cuda_error = any(token in message for token in ("cuda", "cudnn", "cublas", "out of memory"))
                if device != "Auto" or plan.device != "cuda" or not cuda_error:
                    raise
                self.events.put(("retry", str(error)))
                cpu_plan = choose_plan(hardware, model, language, "CPU")
                # Preserve any partial results from the first attempt.
                folder = Path(tempfile.mkdtemp(prefix=f"{source.stem[:60]}-cpu-retry-", dir=destination))
                lines = execute(cpu_plan, " | Retried after GPU error")
            self.events.put(("finished", (folder, f"Done. {lines} transcript segments.", False)))
        except Cancelled:
            self.events.put(("finished", (folder, "Cancelled. Any partial results are in the output folder.", False)))
        except Exception as error:
            self.events.put(("finished", (folder, str(error), True)))

    def clear_preview(self):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.configure(state="disabled")

    def cancel(self):
        self.stop.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Cancelling after the current operation...")

    def close(self):
        if self.worker is not None:
            self.cancel()
            self.root.after(200, self.close_when_idle)
        else:
            self.root.destroy()

    def close_when_idle(self):
        if self.worker is None:
            self.root.destroy()
        else:
            self.root.after(200, self.close_when_idle)

    def poll(self):
        try:
            for _ in range(150):
                kind, value = self.events.get_nowait()
                if kind == "plan":
                    self.details.set(value)
                    if not self.stop.is_set():
                        self.status.set("Loading model. First use may download model files...")
                elif kind == "retry":
                    self.clear_preview()
                    self.details.set(f"GPU retry on CPU: {value}")
                    self.bar.configure(value=0)
                elif kind == "progress":
                    elapsed, total, language, line = value
                    if total and self.bar.cget("mode") != "determinate":
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                    if total:
                        self.bar.configure(value=max(float(self.bar["value"]), min(99, elapsed / total * 100)))
                    if not self.stop.is_set():
                        self.status.set(f"Transcribing  {timecode(elapsed, '.')}" +
                                        (f" / {timecode(total, '.')}" if total else "") +
                                        (f"  |  Language: {language}" if language else ""))
                    if line:
                        self.preview.configure(state="normal")
                        self.preview.insert("end", f"[{timecode(elapsed, '.')}] {line}\n")
                        if int(self.preview.index("end-1c").split(".")[0]) > 500:
                            self.preview.delete("1.0", "101.0")
                        self.preview.see("end")
                        self.preview.configure(state="disabled")
                elif kind == "finished":
                    self.output, status, error = value
                    self.bar.stop()
                    self.bar.configure(mode="determinate", value=100 if status.startswith("Done.") else 0)
                    self.status.set("Transcription failed." if error else status)
                    self.worker = None
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    for field in self.settings:
                        field.configure(state="readonly" if isinstance(field, ttk.Combobox) else "normal")
                    if self.output:
                        self.output_button.configure(state="normal")
                    if error:
                        messagebox.showerror("Transcription failed", status, parent=self.root)
        except Empty:
            pass
        self.root.after(100, self.poll)


if __name__ == "__main__":
    root = tk.Tk()
    Transcriber(root)
    if sys.argv[1:] == ["--smoke-test"]:
        import av
        import ctranslate2
        from faster_whisper.vad import get_vad_model
        from sherpa_onnx import onnxruntime_version

        root.withdraw()
        root.update()
        av.Codec("mp3", "r")
        av.Codec("aac", "r")
        if "int8" not in ctranslate2.get_supported_compute_types("cpu"):
            raise RuntimeError("The packaged CPU runtime does not support int8.")
        if not onnxruntime_version:
            raise RuntimeError("The packaged speaker runtime could not be identified.")
        get_vad_model()
        root.destroy()
    else:
        root.mainloop()

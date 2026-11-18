from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

datas = collect_data_files("faster_whisper")
datas += copy_metadata("faster-whisper", recursive=True)
for distribution in distributions():
    for file in distribution.files or ():
        if file.parts[0].endswith(".dist-info") and (
            "licenses" in file.parts[1:-1]
            or file.name.casefold().startswith(("license", "copying", "notice", "authors"))
        ):
            source = Path(distribution.locate_file(file))
            if source.is_file():
                datas.append((str(source), str(Path("third-party-notices") / file.parent)))

analysis = Analysis(
    ["app.py"],
    binaries=collect_dynamic_libs("sherpa_onnx"),
    datas=datas,
    excludes=["matplotlib", "pandas", "scipy", "torch"],
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, name="Vid2text", exclude_binaries=True, console=False)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, name="Vid2text")

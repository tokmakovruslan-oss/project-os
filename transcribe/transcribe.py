#!/usr/bin/env python3
"""Дословная расшифровка аудио/видео в текст с посекундными таймкодами.

Использует faster-whisper с пословными таймкодами. Каждое слово привязано
к секунде, в которой оно началось, и результат выводится построчно:

    [00:00:00] Привет, эм, это
    [00:00:01] тестовая запись
    [00:00:03] ну, в общем...

Примеры:
    python transcribe.py lecture.mp3
    python transcribe.py call.ogg --model large-v3 --language ru --empty-seconds
    python transcribe.py video.mp4 --formats txt,srt,json --out-dir out/
"""

import argparse
import json
import sys
from pathlib import Path

# Подсказка, которая удерживает Whisper от «причёсывания» речи:
# сохраняет слова-паразиты, повторы и оговорки.
VERBATIM_PROMPTS = {
    "ru": "Эм, ну, значит... это, как бы, вот. Да-да, я, я имею в виду, ну, э-э...",
    "en": "Um, uh, so... like, you know, I mean, I, I was, uh, hmm.",
}


def fmt_ts(seconds, srt=False):
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(path, model_name, language, device, compute_type, verbatim):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    prompt = VERBATIM_PROMPTS.get(language or "ru") if verbatim else None
    segments, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        initial_prompt=prompt,
        # Без опоры на предыдущий текст модель меньше «додумывает» и не зацикливается.
        condition_on_previous_text=False,
        beam_size=5,
        vad_filter=False,  # не выкидываем тихие куски — нужна каждая секунда
    )

    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word,
                "prob": round(w.probability, 3),
            })
        print(f"\r  обработано: {fmt_ts(seg.end)} / {fmt_ts(info.duration)}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return words, info


def group_by_second(words, duration, include_empty):
    buckets = {}
    for w in words:
        buckets.setdefault(int(w["start"]), []).append(w["word"])
    last = int(duration) if include_empty else (max(buckets) if buckets else -1)
    lines = []
    for sec in range(0, last + 1):
        text = "".join(buckets.get(sec, [])).strip()
        if text or include_empty:
            lines.append((sec, text))
    return lines


def to_txt(lines):
    return "\n".join(f"[{fmt_ts(s)}] {t}".rstrip() for s, t in lines) + "\n"


def to_srt(lines):
    out = []
    for i, (s, t) in enumerate((l for l in lines if l[1]), 1):
        out.append(f"{i}\n{fmt_ts(s, True)} --> {fmt_ts(s + 1, True)}\n{t}\n")
    return "\n".join(out)


def to_json(lines, words, info):
    return json.dumps({
        "language": info.language,
        "duration": round(info.duration, 3),
        "seconds": [{"second": s, "time": fmt_ts(s), "text": t} for s, t in lines],
        "words": words,
    }, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Дословная посекундная расшифровка аудио")
    p.add_argument("audio", type=Path, help="аудио/видео файл (mp3, wav, ogg, m4a, mp4, ...)")
    p.add_argument("--model", default="medium",
                   help="модель Whisper: tiny, base, small, medium, large-v3 (по умолчанию medium)")
    p.add_argument("--language", default="ru", help="язык речи; 'auto' — автоопределение")
    p.add_argument("--device", default="auto", help="cpu, cuda или auto")
    p.add_argument("--compute-type", default="auto", help="int8, float16, ... (по умолчанию auto)")
    p.add_argument("--formats", default="txt,srt,json", help="через запятую: txt, srt, json")
    p.add_argument("--out-dir", type=Path, help="куда сохранить (по умолчанию рядом с файлом)")
    p.add_argument("--empty-seconds", action="store_true",
                   help="выводить и пустые секунды (тишину), чтобы была строка на каждую секунду")
    p.add_argument("--no-verbatim", action="store_true",
                   help="не подталкивать модель сохранять слова-паразиты и повторы")
    args = p.parse_args()

    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")
    language = None if args.language == "auto" else args.language

    print(f"Модель {args.model}, файл {args.audio.name}", file=sys.stderr)
    words, info = transcribe(args.audio, args.model, language, args.device,
                             args.compute_type, not args.no_verbatim)
    lines = group_by_second(words, info.duration, args.empty_seconds)

    out_dir = args.out_dir or args.audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "txt": lambda: to_txt(lines),
        "srt": lambda: to_srt(lines),
        "json": lambda: to_json(lines, words, info),
    }
    for fmt in (f.strip() for f in args.formats.split(",") if f.strip()):
        if fmt not in renderers:
            sys.exit(f"Неизвестный формат: {fmt}")
        dest = out_dir / f"{args.audio.stem}.{fmt}"
        dest.write_text(renderers[fmt](), encoding="utf-8")
        print(f"Сохранено: {dest}", file=sys.stderr)

    sys.stdout.write(to_txt(lines))


if __name__ == "__main__":
    main()

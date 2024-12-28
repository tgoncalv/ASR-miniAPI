import argparse
import os
import time

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcription.txt",
        help="Path to output transcription file",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting output file if it already exists",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Model ID. Check https://huggingface.co/models?pipeline_tag=automatic-speech-recognition for more models.",
    )
    parser.add_argument(
        "--hide_timestamps",
        action="store_true",
        help="Deactivate timestamps in transcription.",
    )
    parser.add_argument(
        "--show_hours",
        action="store_true",
        help="Show hours in timestamp.",
    )
    # check if arguments are valid
    res = parser.parse_args()
    if res.hide_timestamps and res.show_hours:
        raise ValueError("Cannot hide timestamps and show hours at the same time.")
    if os.path.exists(res.output) and not res.allow_overwrite:
        raise ValueError(
            f"Output file {res.output} already exists. Use --allow_overwrite to overwrite it."
        )
    return res


def format_timestamp(timestamp: float, show_hours: bool = False) -> str:
    seconds = int(timestamp % 60)
    if show_hours:
        minutes = int((timestamp % 3600) // 60)
        hours = int(timestamp // 3600)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    minutes = int(timestamp // 60)
    return f"{minutes:02}:{seconds:02}"


def main(args):
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Load an processed audio
    print(f"Preprocessing audio file '{args.input}'...")
    _time = time.time()
    audio, sample_rate = librosa.load(args.input, sr=16000)
    inputs = processor(
        audio,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=sample_rate,
    ).to(device)
    print(f"Preprocessing took {time.time() - _time:.2f}s")
    # Generate transcription with timestamps
    print("Generating transcription...")
    _time = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(
            **inputs,
            task="transcribe",
            return_timestamps=True,
            return_segments=True,
            num_beams=5,
            length_penalty=1.0,
        )
    print(f"Transcription took {time.time() - _time:.2f}s")
    # Save transcription with timestamps to file
    print(f"Saving transcription to '{args.output}'...")
    _time = time.time()
    with open(args.output, "w") as f:
        for segment in predicted_ids["segments"][0]:
            start_time = segment["start"].item()
            end_time = segment["end"].item()
            if not args.hide_timestamps:
                start_time = format_timestamp(start_time, args.show_hours)
                end_time = format_timestamp(end_time, args.show_hours)
            text = processor.decode(segment["tokens"], skip_special_tokens=True)
            f.write(f"[{start_time} - {end_time}] {text}")
            f.write("\n")
    print(f"Saving took {time.time() - _time:.2f}s")


if __name__ == "__main__":
    args = get_args()
    main(args)

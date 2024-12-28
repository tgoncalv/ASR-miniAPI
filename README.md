# ASR-miniAPI

Minimalist API implementation for Automatic Speech Recognition (ASR) using Whisper models from Hugging Face.

Author: Taïga Gonçalves (https://tgoncalv.github.io/)

---

## Installation and Setup

### Step 1: Download the project:

Clone the repository in the folder of your choice:

```
git clone https://github.com/tgoncalv/ASR-miniAPI.git
cd ASR-miniAPI
```

Alternatively, download the ZIP file:

- Go to the repository page: https://github.com/tgoncalv/ASR-miniAPI
- Click on the green "Code" button and select "Download ZIP".
- Extract the ZIP file and navigate to the extracted folder.

#### Step 2. Create a virtual environment (recommended):

- **Linux/MacOS**:
```
python3 -m venv venv
source venv/bin/activate
```

- **Windows**:
```
python -m venv venv
venv\Scripts\activate
```

#### Step 3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

To transcribe an audio file, run the following command:

```
python main.py --input <path_to_audio_file> --output <output_file> [OPTIONS]
```

### Options

- `--input <path_to_audio_file>`: **(Required)** Path to the input audio file to be transcribed.

- `--output <path_to_output_file>`: **(Optional)** Path to the output file where the transcription will be saved. Default: `transcription.txt`.

- `--allow_overwrite`: **(Optional)** Allows overwriting the output file if it already exists. Without this option, the program will raise an error if the file exists.

- `--model_id <model_name>`: **(Optional)** Specifies the Whisper model to be used. Default: `openai/whisper-large-v3-turbo`. For more models, check [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition).

- `--hide_timestamps`: **(Optional)** Hides timestamps in the transcription output.

- `--show_hours`: **(Optional)** Displays timestamps in `[hh:mm:ss]` format, even if the transcription is less than an hour long. This option is ignored if `--hide_timestamps` is also set.

---

### Examples

1. Basic Transcription:

```
python main.py --input audio.mp3 --output transcription.txt
```

2. Overwrite Existing Output:

```
python main.py --input audio.mp3 --output transcription.txt --allow_overwrite
```

3. Customize Timestamps:

   - Hide timestamps:

   ```
   python main.py --input audio.mp3 --output transcription.txt --hide_timestamps
   ```

   - Show hours in timestamps:

   ```
   python main.py --input audio.mp3 --output transcription.txt --show_hours
   ```

4. Use a Specific Model:

```
python main.py --input audio.mp3 --output transcription.txt --model_id openai/whisper-large-v3
```

---

## Popular Models

Check Hugging Face ASR Models (https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) for more models. Popular options:

- openai/whisper-large-v3-turbo (Default)
- openai/whisper-large-v3

---

## License

This project is licensed under the MIT License.

# Low-Resource Speech Translation Data Pipeline

This project automates the collection, preprocessing, and organization of speech and text datasets for low-resource African languages (e.g., Hausa, Amharic, Somali, Swahili, Yoruba, Zulu, etc.).

## Features
- **Automated Download**: Fetches datasets from Hugging Face.
- **Audio Preprocessing**: Resamples audio to 16kHz, converts to mono, and filters by duration.
- **Text Preprocessing**: Cleans and normalizes text.
- **Data Merging**: Aggregates processed datasets into a single unified dataset.
- **Hugging Face Integration**: Pushes processed/merged data to the Hub.

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Aman-byte1/speech_IWSLT.git
   cd speech_IWSLT
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `torchcodec`, the pipeline leverages `librosa` and `soundfile` as fallbacks.*

3. **Configure**
   - Edit `config.yaml` to set your Hugging Face token (`hf_token`) and target organization (`output_org`).

## Usage

### Run Full Pipeline (Process, Merge, Push)
```bash
python data.py --config config.yaml --lang hausa
```
*Replace `hausa` with your target language or omit `--lang` to process all configured datasets.*

### Merge Only
If you have already processed data and just want to merge and push:
```bash
python data.py --config config.yaml --merge_only
```

## Deployment on RunPod (or Remote Servers)

### Git Authentication
GitHub no longer supports password authentication. To clone this repository on a remote server like RunPod, use a **Personal Access Token (PAT)**.

1. **Generate a Token**:
   - Go to [GitHub Developer Settings](https://github.com/settings/tokens).
   - Generate a **New Token (Classic)** with `repo` scope.
   - Copy the token (starts with `ghp_`).

2. **Clone with Token**:
   ```bash
   git clone https://<YOUR_TOKEN>@github.com/Aman-byte1/speech_IWSLT.git
   ```

3. **Install & Run**:
   Follow the Setup instructions above.

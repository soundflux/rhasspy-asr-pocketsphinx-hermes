# Rhasspy ASR Pocketsphinx Hermes MQTT Service

Implements `hermes/asr` functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using [rhasspy-asr-pocketsphinx](https://github.com/synesthesiam/rhasspy-asr-pocketsphinx).

## Running With Docker

```bash
docker run -it rhasspy/rhasspy-asr-pocketsphinx-hermes:<VERSION> <ARGS>
```

## Building From Source

Make sure you have the required dependencies:

```bash
sudo apt-get update
sudo apt-get install build-essential
```

Clone the repository and create the virtual environment:

```bash
git clone https://github.com/rhasspy/rhasspy-asr-pocketsphinx-hermes.git
cd rhasspy-asr-pocketsphinx-hermes
make venv
```

Run the `bin/rhasspy-asr-pocketsphinx-hermes` script to access the command-line interface:

```bash
bin/rhasspy-asr-pocketsphinx-hermes --help
```

## Building the Debian Package

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make debian
```

If successful, you'll find a `.deb` file in the `dist` directory that can be installed with `apt`.

## Building the Docker Image

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make docker
```

This will create a Docker image tagged `rhasspy/rhasspy-asr-pocketsphinx-hermes:<VERSION>` where `VERSION` comes from the file of the same name in the source root directory.

## Command-Line Options

```
usage: rhasspy-asr-pocketsphinx-hermes [-h] --acoustic-model ACOUSTIC_MODEL
                                       --dictionary DICTIONARY
                                       --language-model LANGUAGE_MODEL
                                       [--mllr-matrix MLLR_MATRIX]
                                       [--base-dictionary BASE_DICTIONARY]
                                       [--reload RELOAD] [--host HOST]
                                       [--port PORT] [--siteId SITEID]
                                       [--stdin-files] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --acoustic-model ACOUSTIC_MODEL
                        Path to Pocketsphinx acoustic model directory (hmm)
  --dictionary DICTIONARY
                        Path to pronunciation dictionary file
  --language-model LANGUAGE_MODEL
                        Path to ARPA language model file
  --mllr-matrix MLLR_MATRIX
                        Path to tuned MLLR matrix file
  --base-dictionary BASE_DICTIONARY
                        Path(s) to base pronunciation dictionary file(s)
  --reload RELOAD       Poll dictionary/language model for given number of
                        seconds and automatically reload when changed
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --siteId SITEID       Hermes siteId(s) to listen for (default: all)
  --stdin-files         Read WAV file paths from stdin
  --debug               Print DEBUG messages to the console
```

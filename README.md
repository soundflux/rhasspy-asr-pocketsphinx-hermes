# Rhasspy ASR Pocketsphinx Hermes MQTT Service

[![Continous Integration](https://github.com/rhasspy/rhasspy-asr-pocketsphinx-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-asr-pocketsphinx-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-asr-pocketsphinx-hermes.svg)](https://github.com/rhasspy/rhasspy-asr-pocketsphinx-hermes/blob/master/LICENSE)

Implements `hermes/asr` functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using [rhasspy-asr-pocketsphinx](https://github.com/rhasspy/rhasspy-asr-pocketsphinx).

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-asr-pocketsphinx-hermes
$ cd rhasspy-asr-pocketsphinx-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-asr-pocketsphinx-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-asr-pocketsphinx-hermes [-h] --acoustic-model ACOUSTIC_MODEL
                                       --dictionary DICTIONARY
                                       [--dictionary-casing {upper,lower,ignore}]
                                       --language-model LANGUAGE_MODEL
                                       [--mllr-matrix MLLR_MATRIX]
                                       [--base-dictionary BASE_DICTIONARY]
                                       [--g2p-model G2P_MODEL]
                                       [--g2p-casing {upper,lower,ignore}]
                                       [--unknown-words UNKNOWN_WORDS]
                                       [--no-overwrite-train]
                                       [--intent-graph INTENT_GRAPH]
                                       [--base-language-model-fst BASE_LANGUAGE_MODEL_FST]
                                       [--base-language-model-weight BASE_LANGUAGE_MODEL_WEIGHT]
                                       [--mixed-language-model-fst MIXED_LANGUAGE_MODEL_FST]
                                       [--voice-skip-seconds VOICE_SKIP_SECONDS]
                                       [--voice-min-seconds VOICE_MIN_SECONDS]
                                       [--voice-speech-seconds VOICE_SPEECH_SECONDS]
                                       [--voice-silence-seconds VOICE_SILENCE_SECONDS]
                                       [--voice-before-seconds VOICE_BEFORE_SECONDS]
                                       [--voice-sensitivity {1,2,3}]
                                       [--host HOST] [--port PORT]
                                       [--username USERNAME]
                                       [--password PASSWORD] [--tls]
                                       [--tls-ca-certs TLS_CA_CERTS]
                                       [--tls-certfile TLS_CERTFILE]
                                       [--tls-keyfile TLS_KEYFILE]
                                       [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                                       [--tls-version TLS_VERSION]
                                       [--tls-ciphers TLS_CIPHERS]
                                       [--site-id SITE_ID] [--debug]
                                       [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --acoustic-model ACOUSTIC_MODEL
                        Path to Pocketsphinx acoustic model directory (hmm)
  --dictionary DICTIONARY
                        Path to read/write pronunciation dictionary file
  --dictionary-casing {upper,lower,ignore}
                        Case transformation for dictionary words (training,
                        default: ignore)
  --language-model LANGUAGE_MODEL
                        Path to read/write ARPA language model file
  --mllr-matrix MLLR_MATRIX
                        Path to read tuned MLLR matrix file
  --base-dictionary BASE_DICTIONARY
                        Path(s) to base pronunciation dictionary file(s)
                        (training)
  --g2p-model G2P_MODEL
                        Phonetisaurus FST model for guessing word
                        pronunciations (training)
  --g2p-casing {upper,lower,ignore}
                        Case transformation for g2p words (training, default:
                        ignore)
  --unknown-words UNKNOWN_WORDS
                        Path to write missing words from dictionary (training)
  --no-overwrite-train  Don't overwrite dictionary/language model during
                        training
  --intent-graph INTENT_GRAPH
                        Path to intent graph (gzipped pickle)
  --base-language-model-fst BASE_LANGUAGE_MODEL_FST
                        Path to base language model FST (training, mixed)
  --base-language-model-weight BASE_LANGUAGE_MODEL_WEIGHT
                        Weight to give base langauge model (training, mixed)
  --mixed-language-model-fst MIXED_LANGUAGE_MODEL_FST
                        Path to write mixed langauge model FST (training,
                        mixed)
  --voice-skip-seconds VOICE_SKIP_SECONDS
                        Seconds of audio to skip before a voice command
  --voice-min-seconds VOICE_MIN_SECONDS
                        Minimum number of seconds for a voice command
  --voice-speech-seconds VOICE_SPEECH_SECONDS
                        Consecutive seconds of speech before start
  --voice-silence-seconds VOICE_SILENCE_SECONDS
                        Consecutive seconds of silence before stop
  --voice-before-seconds VOICE_BEFORE_SECONDS
                        Seconds to record before start
  --voice-sensitivity {1,2,3}
                        VAD sensitivity (1-3)
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```

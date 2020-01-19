"""Command-line interface to rhasspyasr-asr-pocketsphinx-hermes"""
import argparse
import json
import logging
import os
import sys
import threading
import time
import typing
from pathlib import Path

import attr
import paho.mqtt.client as mqtt

from rhasspyasr_pocketsphinx import PocketsphinxTranscriber
from rhasspyhermes.asr import AsrTextCaptured

from . import AsrHermesMqtt

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-pocketsphinx-hermes")
    parser.add_argument(
        "--acoustic-model",
        required=True,
        help="Path to Pocketsphinx acoustic model directory (hmm)",
    )
    parser.add_argument(
        "--dictionary", required=True, help="Path to pronunciation dictionary file"
    )
    parser.add_argument(
        "--language-model", required=True, help="Path to ARPA language model file"
    )
    parser.add_argument(
        "--mllr-matrix", default=None, help="Path to tuned MLLR matrix file"
    )
    parser.add_argument(
        "--base-dictionary",
        action="append",
        help="Path(s) to base pronunciation dictionary file(s)",
    )
    parser.add_argument(
        "--reload",
        type=float,
        default=None,
        help="Poll dictionary/language model for given number of seconds and automatically reload when changed",
    )
    parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )
    parser.add_argument(
        "--stdin-files", action="store_true", help="Read WAV file paths from stdin"
    )
    parser.add_argument(
        "--g2p-model", help="Phonetisaurus FST model for guessing word pronunciations"
    )
    parser.add_argument(
        "--train",
        help="Generate dictionary and language model from intent JSON graph file",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    try:
        # Load transciber
        _LOGGER.debug(
            "Loading Pocketsphinx decoder with (hmm=%s, dict=%s, lm=%s, mllr=%s)",
            args.acoustic_model,
            args.dictionary,
            args.language_model,
            args.mllr_matrix,
        )

        # Convert to paths
        args.acoustic_model = Path(args.acoustic_model)
        args.dictionary = Path(args.dictionary)
        args.language_model = Path(args.language_model)

        if args.mllr_matrix:
            args.mllr_matrix = Path(args.mllr_matrix)

        if args.base_dictionary:
            args.base_dictionary = [Path(p) for p in args.base_dictionary]

        if args.train:
            args.train = Path(args.train)

        if args.g2p_model:
            args.g2p_model = Path(args.g2p_model)

        transcriber = PocketsphinxTranscriber(
            args.acoustic_model,
            args.dictionary,
            args.language_model,
            mllr_matrix=args.mllr_matrix,
            debug=args.debug,
        )

        if args.stdin_files:
            # Read WAV file(s) from stdin
            hermes = AsrHermesMqtt(client=None, transcriber=transcriber)

            for wav_path in sys.stdin:
                wav_path = Path(wav_path.strip())
                _LOGGER.debug("Transcribing %s", str(wav_path))
                wav_bytes = wav_path.read_bytes()
                audio_bytes = hermes.maybe_convert_wav(wav_bytes)
                result = hermes.transcribe(audio_bytes)

                # Print messages as JSON
                assert isinstance(result, AsrTextCaptured)
                json.dump(attr.asdict(result), sys.stdout)
                print("")
            return
        elif args.train:
            if not args.base_dictionary:
                _LOGGER.warning(
                    "You probably want to pass a base dictionary when re-training."
                )

            if not args.g2p_model:
                _LOGGER.warning(
                    "You probably want to pass a g2p model when re-training."
                )

            # Re-train ASR system
            hermes = AsrHermesMqtt(
                client=None,
                transcriber=transcriber,
                base_dictionaries=args.base_dictionary,
                g2p_model=args.g2p_model,
            )

            _LOGGER.debug("Re-training from %s", args.train)
            with open(args.train, "r") as json_file:
                json_graph = json.load(json_file)
                hermes.retrain(json_graph)
            return

        # Listen for messages
        client = mqtt.Client()
        hermes = AsrHermesMqtt(
            client,
            transcriber,
            base_dictionaries=args.base_dictionary,
            siteIds=args.siteId,
        )

        if args.reload:
            # Start polling thread
            threading.Thread(
                target=poll_files,
                args=(
                    args.reload,
                    args.acoustic_model,
                    args.dictionary,
                    args.language_model,
                    hermes,
                    args.mllr_matrix,
                    args.debug,
                ),
                daemon=True,
            ).start()

        def on_disconnect(client, userdata, flags, rc):
            try:
                # Automatically reconnect
                _LOGGER.info("Disconnected. Trying to reconnect...")
                client.reconnect()
            except Exception:
                _LOGGER.exception("on_disconnect")

        # Connect
        client.on_connect = hermes.on_connect
        client.on_message = hermes.on_message
        client.on_disconnect = on_disconnect

        _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
        client.connect(args.host, args.port)

        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


# -----------------------------------------------------------------------------


def poll_files(
    seconds: float,
    acoustic_model_dir: Path,
    dictionary_path: Path,
    language_model_path: Path,
    hermes: AsrHermesMqtt,
    mllr_matrix_path: typing.Optional[Path] = None,
    debug: bool = False,
):
    """Poll dictionary/language and re-load transcriber when changed."""
    last_timestamp: typing.Dict[Path, int] = {
        dictionary_path: 0,
        language_model_path: 0,
    }

    while True:
        time.sleep(seconds)
        try:
            just_reloaded = False
            for path in [dictionary_path, language_model_path]:
                timestamp = os.stat(path).st_mtime_ns
                if timestamp != last_timestamp:
                    if not just_reloaded:
                        # Reload transcriber
                        _LOGGER.debug(
                            "Loading Pocketsphinx decoder with (hmm=%s, dict=%s, lm=%s, mllr=%s)",
                            acoustic_model_dir,
                            dictionary_path,
                            language_model_path,
                            mllr_matrix_path,
                        )

                        # Set in hermes object
                        hermes.transcriber = PocketsphinxTranscriber(
                            acoustic_model_dir,
                            dictionary_path,
                            language_model_path,
                            mllr_matrix=mllr_matrix_path,
                            debug=debug,
                        )

                        # Avoid reloading if more than one path changed
                        just_reloaded = True

                    # Update timestamp
                    last_timestamp[path] = timestamp
        except Exception:
            _LOGGER.exception("poll_files")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

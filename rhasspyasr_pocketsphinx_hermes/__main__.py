"""Command-line interface to rhasspyasr-asr-pocketsphinx-hermes"""
import argparse
import json
import logging
import os
import threading
import time
import typing
from pathlib import Path
from uuid import uuid4

import paho.mqtt.client as mqtt
import rhasspyasr_pocketsphinx
from rhasspyasr_pocketsphinx import PocketsphinxTranscriber
from rhasspyhermes.asr import AsrTrain

from . import AsrHermesMqtt

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Dispatch to appropriate sub-command
    args.func(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-pocketsphinx-hermes")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # Run settings
    run_parser = sub_parsers.add_parser("run", help="Run MQTT service")
    run_parser.set_defaults(func=run_mqtt)

    run_parser.add_argument(
        "--acoustic-model",
        required=True,
        help="Path to Pocketsphinx acoustic model directory (hmm)",
    )
    run_parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to read/write pronunciation dictionary file",
    )
    run_parser.add_argument(
        "--dictionary-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for dictionary words (training, default: ignore)",
    )
    run_parser.add_argument(
        "--language-model",
        required=True,
        help="Path to read/write ARPA language model file",
    )
    run_parser.add_argument(
        "--mllr-matrix", default=None, help="Path to read tuned MLLR matrix file"
    )
    run_parser.add_argument(
        "--watch-delay",
        type=float,
        default=1.0,
        help="Seconds between polling intent graph file for training",
    )
    run_parser.add_argument(
        "--intent-graph", help="Path to write intent graph JSON file (training)"
    )

    run_parser.add_argument(
        "--base-dictionary",
        action="append",
        help="Path(s) to base pronunciation dictionary file(s) (training)",
    )
    run_parser.add_argument(
        "--g2p-model",
        help="Phonetisaurus FST model for guessing word pronunciations (training)",
    )
    run_parser.add_argument(
        "--g2p-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for g2p words (training, default: ignore)",
    )

    # MQTT settings (run)
    run_parser.add_argument(
        "--host", default="localhost", help="MQTT host (default: localhost)"
    )
    run_parser.add_argument(
        "--port", type=int, default=1883, help="MQTT port (default: 1883)"
    )
    run_parser.add_argument(
        "--siteId",
        action="append",
        help="Hermes siteId(s) to listen for (default: all)",
    )

    # -------------------------------------------------------------------------

    # Train settings
    train_parser = sub_parsers.add_parser(
        "train", help="Generate dictionary/language model from intent graph and exit"
    )
    train_parser.set_defaults(func=train)

    train_parser.add_argument(
        "--intent-graph", required=True, help="Path to read intent graph JSON file"
    )

    train_parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to write pronunciation dictionary file",
    )
    train_parser.add_argument(
        "--language-model", required=True, help="Path to write ARPA language model file"
    )

    train_parser.add_argument(
        "--base-dictionary",
        action="append",
        required=True,
        help="Path(s) to base pronunciation dictionary file(s)",
    )
    train_parser.add_argument(
        "--g2p-model", help="Phonetisaurus FST model for guessing word pronunciations"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
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

        if args.g2p_model:
            args.g2p_model = Path(args.g2p_model)

        if args.intent_graph:
            args.intent_graph = Path(args.intent_graph)

        def make_transcriber():
            return PocketsphinxTranscriber(
                args.acoustic_model,
                args.dictionary,
                args.language_model,
                mllr_matrix=args.mllr_matrix,
                debug=args.debug,
            )

        # Listen for messages
        client = mqtt.Client()
        hermes = AsrHermesMqtt(
            client,
            make_transcriber,
            dictionary=args.dictionary,
            language_model=args.language_model,
            base_dictionaries=args.base_dictionary,
            siteIds=args.siteId,
            dictionary_word_transform=get_word_transform(args.dictionary_casing),
            g2p_model=args.g2p_model,
            g2p_word_transform=get_word_transform(args.g2p_casing),
        )

        if args.intent_graph and (args.watch_delay > 0):
            _LOGGER.debug(
                "Watching %s for changes (every %s second(s))",
                str(args.intent_graph),
                args.watch_delay,
            )

            # Start polling thread
            threading.Thread(
                target=poll_files,
                args=(args.intent_graph, args.watch_delay, hermes, args.debug),
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


def poll_files(
    graph_path: Path, delay_seconds: float, hermes: AsrHermesMqtt, debug: bool = False
):
    """Poll intent graph and re-trains when changed."""
    last_timestamps: typing.Dict[Path, int] = {}

    while True:
        time.sleep(delay_seconds)
        try:
            retrain = False
            for path in [graph_path]:
                timestamp = os.stat(path).st_mtime_ns
                last_timestamp = last_timestamps.get(path)

                if (last_timestamp is not None) and (timestamp != last_timestamp):
                    retrain = True

                # Update timestamp
                last_timestamps[path] = timestamp

            if retrain:
                _LOGGER.debug("%s changed. Re-training...", str(graph_path))

                with open(graph_path, "r") as graph_file:
                    graph_dict = json.load(graph_file)
                    result = hermes.handle_train(
                        AsrTrain(id=str(uuid4()), graph_dict=graph_dict)
                    )
                    hermes.publish(result)
        except Exception:
            _LOGGER.exception("poll_files")


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------


def train(args: argparse.Namespace):
    """Generates ASR artifacts from intent JSON graph."""
    # Convert to paths
    args.dictionary = Path(args.dictionary)
    args.language_model = Path(args.language_model)

    if args.base_dictionary:
        args.base_dictionary = [Path(p) for p in args.base_dictionary]

    if args.g2p_model:
        args.g2p_model = Path(args.g2p_model)

    if args.intent_graph:
        args.intent_graph = Path(args.intent_graph)

    if not args.g2p_model:
        _LOGGER.warning("You probably want to pass a g2p model when re-training.")

    # Re-train ASR system
    _LOGGER.debug("Re-training from %s", args.intent_graph)
    with open(args.intent_graph, "r") as json_file:
        graph_dict = json.load(json_file)
        rhasspyasr_pocketsphinx.train(
            graph_dict,
            args.dictionary,
            args.language_model,
            args.base_dictionary,
            args.g2p_model,
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

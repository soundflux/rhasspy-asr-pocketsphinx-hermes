"""Command-line interface to rhasspyasr-asr-pocketsphinx-hermes"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
from rhasspyasr_pocketsphinx import PocketsphinxTranscriber

from . import AsrHermesMqtt

_LOGGER = logging.getLogger("rhasspyasr_pocketsphinx_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=args.log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=args.log_format)

    _LOGGER.debug(args)

    run_mqtt(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-pocketsphinx-hermes")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    parser.add_argument(
        "--acoustic-model",
        required=True,
        help="Path to Pocketsphinx acoustic model directory (hmm)",
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to read/write pronunciation dictionary file",
    )
    parser.add_argument(
        "--dictionary-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for dictionary words (training, default: ignore)",
    )
    parser.add_argument(
        "--language-model",
        required=True,
        help="Path to read/write ARPA language model file",
    )
    parser.add_argument(
        "--mllr-matrix", default=None, help="Path to read tuned MLLR matrix file"
    )
    parser.add_argument(
        "--intent-graph", help="Path to write intent graph JSON file (training)"
    )

    parser.add_argument(
        "--base-dictionary",
        action="append",
        help="Path(s) to base pronunciation dictionary file(s) (training)",
    )
    parser.add_argument(
        "--g2p-model",
        help="Phonetisaurus FST model for guessing word pronunciations (training)",
    )
    parser.add_argument(
        "--g2p-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for g2p words (training, default: ignore)",
    )
    parser.add_argument(
        "--unknown-words", help="Path to write missing words from dictionary (training)"
    )
    parser.add_argument(
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite dictionary/language model during training",
    )

    # MQTT settings (run)
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
        "--log-format",
        default="[%(levelname)s:%(asctime)s] %(name)s: %(message)s",
        help="Python logger format",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
    try:
        loop = asyncio.get_event_loop()

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

        if args.unknown_words:
            args.unknown_words = Path(args.unknown_words)

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
            unknown_words=args.unknown_words,
            no_overwrite_train=args.no_overwrite_train,
            loop=loop,
        )

        if args.intent_graph and (args.watch_delay > 0):
            _LOGGER.debug(
                "Watching %s for changes (every %s second(s))",
                str(args.intent_graph),
                args.watch_delay,
            )

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
        client.loop_start()

        # Run event loop
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

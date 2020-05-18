"""Command-line interface to rhasspyasr-asr-pocketsphinx-hermes"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
import rhasspyhermes.cli as hermes_cli
from rhasspyasr_pocketsphinx import PocketsphinxTranscriber

from . import AsrHermesMqtt

_LOGGER = logging.getLogger("rhasspyasr_pocketsphinx_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    run_mqtt(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-pocketsphinx-hermes")
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
    parser.add_argument("--intent-graph", help="Path to intent graph (gzipped pickle)")

    # Mixed language modeling
    parser.add_argument(
        "--base-language-model-fst",
        help="Path to base language model FST (training, mixed)",
    )
    parser.add_argument(
        "--base-language-model-weight",
        type=float,
        default=0,
        help="Weight to give base langauge model (training, mixed)",
    )
    parser.add_argument(
        "--mixed-language-model-fst",
        help="Path to write mixed langauge model FST (training, mixed)",
    )

    # Silence detection
    parser.add_argument(
        "--voice-skip-seconds",
        type=float,
        default=0.0,
        help="Seconds of audio to skip before a voice command",
    )
    parser.add_argument(
        "--voice-min-seconds",
        type=float,
        default=1.0,
        help="Minimum number of seconds for a voice command",
    )
    parser.add_argument(
        "--voice-speech-seconds",
        type=float,
        default=0.3,
        help="Consecutive seconds of speech before start",
    )
    parser.add_argument(
        "--voice-silence-seconds",
        type=float,
        default=0.5,
        help="Consecutive seconds of silence before stop",
    )
    parser.add_argument(
        "--voice-before-seconds",
        type=float,
        default=0.5,
        help="Seconds to record before start",
    )
    parser.add_argument(
        "--voice-sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="VAD sensitivity (1-3)",
    )

    hermes_cli.add_hermes_args(parser)

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
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

    if args.base_language_model_fst:
        args.base_language_model_fst = Path(args.base_language_model_fst)

    if args.mixed_language_model_fst:
        args.mixed_language_model_fst = Path(args.mixed_language_model_fst)

    def make_transcriber(language_model: Path):
        return PocketsphinxTranscriber(
            args.acoustic_model,
            args.dictionary,
            language_model,
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
        dictionary_word_transform=get_word_transform(args.dictionary_casing),
        g2p_model=args.g2p_model,
        g2p_word_transform=get_word_transform(args.g2p_casing),
        unknown_words=args.unknown_words,
        no_overwrite_train=args.no_overwrite_train,
        intent_graph_path=args.intent_graph,
        base_language_model_fst=args.base_language_model_fst,
        base_language_model_weight=args.base_language_model_weight,
        mixed_language_model_fst=args.mixed_language_model_fst,
        skip_seconds=args.voice_skip_seconds,
        min_seconds=args.voice_min_seconds,
        speech_seconds=args.voice_speech_seconds,
        silence_seconds=args.voice_silence_seconds,
        before_seconds=args.voice_before_seconds,
        vad_mode=args.voice_sensitivity,
        site_ids=args.site_id,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()
        hermes.cleanup()


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

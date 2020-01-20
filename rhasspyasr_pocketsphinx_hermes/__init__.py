"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import io
import json
import logging
import shutil
import subprocess
import tempfile
import typing
import wave
from pathlib import Path

import attr
import rhasspyasr.utils
import rhasspynlu
from rhasspyasr_pocketsphinx import PocketsphinxTranscriber
from rhasspyhermes.asr import (
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

from .messages import AsrError, AsrTrain, AsrTrainSuccess

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class MissingWordPronunciationsException(Exception):
    """Raised when missing word pronunciations and no g2p model."""

    def __init__(self, words: typing.List[str]):
        super().__init__(self)
        self.words = words

    def __repr__(self):
        return f"Missing pronunciations for: {self.words}"


@attr.s
class SessionInfo:
    """Information about an open session."""

    sessionId: str = attr.ib()
    recorder: VoiceCommandRecorder = attr.ib()
    transcription_sent: bool = attr.ib(default=False)


# -----------------------------------------------------------------------------


class AsrHermesMqtt:
    """Hermes MQTT server for Rhasspy ASR using Pocketsphinx."""

    def __init__(
        self,
        client,
        transcriber: PocketsphinxTranscriber,
        base_dictionaries: typing.Optional[typing.List[Path]] = None,
        g2p_model: typing.Optional[Path] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
    ):
        self.client = client
        self.transcriber = transcriber
        self.base_dictionaries = base_dictionaries or []
        self.g2p_model = g2p_model
        self.siteIds = siteIds or []
        self.enabled = enabled

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(max_seconds=None)

        self.make_recorder = make_recorder or default_recorder

        # WAV buffers for each session
        self.sessions: typing.Dict[str, SessionInfo] = {}

        # Topic to listen for WAV chunks on
        self.audioframe_topics: typing.List[str] = []
        for siteId in self.siteIds:
            self.audioframe_topics.append(AudioFrame.topic(siteId=siteId))

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def start_listening(self, message: AsrStartListening):
        """Start recording audio data for a session."""
        session = self.sessions.get(message.sessionId)
        if not session:
            session = SessionInfo(
                sessionId=message.sessionId, recorder=self.make_recorder(),
            )
            self.sessions[message.sessionId] = session

        # Start session
        assert session
        session.recorder.start()
        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    def stop_listening(
        self, message: AsrStopListening
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AsrError]]:
        """Stop recording audio data for a session."""
        try:
            session = self.sessions.pop(message.sessionId, None)
            if session:
                # Stop session
                audio_data = session.recorder.stop()
                if not session.transcription_sent and audio_data:
                    # Send transcription
                    session.transcription_sent = True

                    yield self.transcribe(
                        audio_data, siteId=message.siteId, sessionId=message.sessionId
                    )

            _LOGGER.debug("Stopping listening (sessionId=%s)", message.sessionId)
        except Exception as e:
            _LOGGER.exception("stop_listening")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=message.siteId,
                sessionId=message.sessionId,
            )

    def handle_audio_frame(
        self, wav_bytes: bytes, siteId: str = "default"
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AsrError]]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(wav_bytes)

        # Add to every open session
        for sessionId, session in self.sessions.items():
            try:
                command = session.recorder.process_chunk(audio_data)
                if command and (command.result == VoiceCommandResult.SUCCESS):
                    assert command.audio_data is not None
                    _LOGGER.debug(
                        "Voice command recorded for session %s (%s byte(s))",
                        sessionId,
                        len(command.audio_data),
                    )

                    session.transcription_sent = True

                    yield self.transcribe(
                        command.audio_data, siteId=siteId, sessionId=sessionId
                    )

                    # Reset session (but keep open)
                    session.recorder.stop()
                    session.recorder.start()
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(self.transcriber),
                    siteId=siteId,
                    sessionId=sessionId,
                )

    def transcribe(
        self, audio_data: bytes, siteId: str = "default", sessionId: str = ""
    ) -> typing.Union[AsrTextCaptured, AsrError]:
        """Transcribe audio data and publish captured text."""
        try:
            with io.BytesIO() as wav_buffer:
                wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
                with wav_file:
                    wav_file.setframerate(self.sample_rate)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setnchannels(self.channels)
                    wav_file.writeframesraw(audio_data)

                transcription = self.transcriber.transcribe_wav(wav_buffer.getvalue())
                if transcription:
                    # Actual transcription
                    return AsrTextCaptured(
                        text=transcription.text,
                        likelihood=transcription.likelihood,
                        seconds=transcription.transcribe_seconds,
                        siteId=siteId,
                        sessionId=sessionId,
                    )

                _LOGGER.warning("Received empty transcription")

                # Empty transcription
                return AsrTextCaptured(
                    text="", likelihood=0, seconds=0, siteId=siteId, sessionId=sessionId
                )
        except Exception as e:
            _LOGGER.exception("transcribe")
            return AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=sessionId,
            )

    # -------------------------------------------------------------------------

    def train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.Union[AsrTrainSuccess, AsrError]:
        """Re-generates language model and dictionary from intent graph"""
        try:
            graph = rhasspynlu.json_to_graph(train.graph_dict)

            # Generate counts
            intent_counts = rhasspynlu.get_intent_ngram_counts(graph)

            # pylint: disable=W0511
            # TODO: Balance counts

            # Use mitlm to create language model
            vocabulary: typing.Set[str] = set()

            with tempfile.NamedTemporaryFile(mode="w") as lm_file:

                # Create ngram counts
                with tempfile.NamedTemporaryFile(mode="w") as count_file:
                    for intent_name in intent_counts:
                        for ngram, count in intent_counts[intent_name].items():
                            # word [word] ... <TAB> count
                            print(*ngram, file=count_file, end="")
                            print("\t", count, file=count_file)

                    count_file.seek(0)
                    with tempfile.NamedTemporaryFile(mode="w+") as vocab_file:
                        ngram_command = [
                            "estimate-ngram",
                            "-order",
                            "3",
                            "-counts",
                            count_file.name,
                            "-write-lm",
                            lm_file.name,
                            "-write-vocab",
                            vocab_file.name,
                        ]

                        _LOGGER.debug(ngram_command)
                        subprocess.check_call(ngram_command)

                        # Extract vocabulary
                        vocab_file.seek(0)
                        for line in vocab_file:
                            line = line.strip()
                            if not line.startswith("<"):
                                vocabulary.add(line)

                # Write dictionary
                if vocabulary:

                    # Load base dictionaries
                    pronunciations: typing.Dict[str, typing.List[str]] = {}

                    for base_dict_path in self.base_dictionaries:
                        _LOGGER.debug("Loading base dictionary from %s", base_dict_path)
                        with open(base_dict_path, "r") as base_dict_file:
                            rhasspyasr.utils.read_dict(
                                base_dict_file, word_dict=pronunciations
                            )

                    with tempfile.NamedTemporaryFile(mode="w") as dict_file:
                        missing_words: typing.Set[str] = set()

                        # Look up each word
                        for word in vocabulary:
                            word_phonemes = pronunciations.get(word)
                            if not word_phonemes:
                                # Add to missing word list
                                _LOGGER.warning("Missing word '%s'", word)
                                missing_words.add(word)
                                continue

                            # Write CMU format
                            for i, phonemes in enumerate(word_phonemes):
                                if i == 0:
                                    print(word, phonemes, file=dict_file)
                                else:
                                    print(f"{word}({i+1})", phonemes, file=dict_file)

                        if missing_words:
                            # Fail if no g2p model is available
                            if not self.g2p_model:
                                raise MissingWordPronunciationsException(
                                    list(missing_words)
                                )

                            # Guess word pronunciations
                            _LOGGER.debug(
                                "Guessing pronunciations for %s", missing_words
                            )
                            with tempfile.NamedTemporaryFile(mode="w") as wordlist_file:
                                # pylint: disable=W0511
                                # TODO: Handle casing
                                for word in missing_words:
                                    print(word, file=wordlist_file)

                                wordlist_file.seek(0)
                                g2p_command = [
                                    "phonetisaurus-apply",
                                    "--model",
                                    str(self.g2p_model),
                                    "--word_list",
                                    wordlist_file.name,
                                    "--nbest",
                                    "1",
                                ]

                                _LOGGER.debug(g2p_command)
                                g2p_lines = subprocess.check_output(
                                    g2p_command, universal_newlines=True
                                ).splitlines()
                                for line in g2p_lines:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        word = parts[0].strip()
                                        phonemes = " ".join(parts[1:]).strip()
                                        print(word, phonemes, file=dict_file)

                        # -----------------------------------------------------

                        # Copy dictionary
                        dict_file.seek(0)
                        shutil.copy(dict_file.name, self.transcriber.dictionary)
                        _LOGGER.debug(
                            "Wrote dictionary to %s", str(self.transcriber.dictionary)
                        )

                # -------------------------------------------------------------

                # Copy language model
                lm_file.seek(0)
                shutil.copy(lm_file.name, self.transcriber.language_model)
                _LOGGER.debug(
                    "Wrote language model to %s", str(self.transcriber.language_model)
                )

            # Force decoder to be reloaded on next use
            self.transcriber.decoder = None

            return AsrTrainSuccess(id=train.id)
        except Exception as e:
            _LOGGER.exception("train")
            return AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=train.id,
            )

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                AsrToggleOn.topic(),
                AsrToggleOff.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
            ]

            if self.audioframe_topics:
                # Specific siteIds
                topics.extend(self.audioframe_topics)
            else:
                # All siteIds
                topics.append(AudioFrame.topic(siteId="+"))

            if self.siteIds:
                # Specific siteIds
                topics.extend(
                    [AsrTrain.topic(siteId=siteId) for siteId in self.siteIds]
                )
            else:
                # All siteIds
                topics.append(AsrTrain.topic(siteId="+"))

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if not msg.topic.endswith("/audioFrame"):
                _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            # Check enable/disable messages
            if msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = True
                    _LOGGER.debug("Enabled")
            elif msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = False
                    _LOGGER.debug("Disabled")

            if not self.enabled:
                # Disabled
                return

            if AudioFrame.is_topic(msg.topic):
                # Check siteId
                if (not self.audioframe_topics) or (
                    msg.topic in self.audioframe_topics
                ):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    siteId = AudioFrame.get_siteId(msg.topic)
                    for result in self.handle_audio_frame(msg.payload, siteId=siteId):
                        self.publish(result)

            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.start_listening(AsrStartListening(**json_payload))
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    for result in self.stop_listening(AsrStopListening(**json_payload)):
                        self.publish(result)
            elif AsrTrain.is_topic(msg.topic):
                # rhasspy/asr/<siteId>/train
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    result = self.train(AsrTrain(**json_payload), siteId=siteId)
                    self.publish(result)
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    # -------------------------------------------------------------------------

    def _convert_wav(self, wav_data: bytes) -> bytes:
        """Converts WAV data to required format with sox. Return raw audio."""
        return subprocess.run(
            [
                "sox",
                "-t",
                "wav",
                "-",
                "-r",
                str(self.sample_rate),
                "-e",
                "signed-integer",
                "-b",
                str(self.sample_width * 8),
                "-c",
                str(self.channels),
                "-t",
                "raw",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            input=wav_data,
        ).stdout

    def maybe_convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format if necessary. Returns raw audio."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                if (
                    (wav_file.getframerate() != self.sample_rate)
                    or (wav_file.getsampwidth() != self.sample_width)
                    or (wav_file.getnchannels() != self.channels)
                ):
                    # Return converted wav
                    return self._convert_wav(wav_bytes)

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())

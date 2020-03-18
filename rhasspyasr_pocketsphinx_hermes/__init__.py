"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import asyncio
import gzip
import io
import json
import logging
import os
import subprocess
import typing
import wave
from collections import defaultdict
from pathlib import Path

import attr
import networkx as nx
import rhasspyasr_pocketsphinx
import rhasspynlu
from rhasspyasr import Transcriber
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspynlu.g2p import PronunciationsType
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger("rhasspyasr_pocketsphinx_hermes")

# -----------------------------------------------------------------------------

TopicArgs = typing.Mapping[str, typing.Any]
GeneratorType = typing.AsyncIterable[
    typing.Union[Message, typing.Tuple[Message, TopicArgs]]
]


@attr.s(auto_attribs=True, slots=True)
class SessionInfo:
    """Information about an open session."""

    sessionId: str
    start_listening: AsrStartListening
    recorder: typing.Optional[VoiceCommandRecorder] = None
    transcription_sent: bool = False
    num_wav_bytes: int = 0
    audio_buffer: typing.Optional[bytes] = None


@attr.s(auto_attribs=True, slots=True)
class PronunciationDictionary:
    """Details of a phonetic dictionary."""

    path: Path
    pronunciations: PronunciationsType = {}
    mtime_ns: typing.Optional[int] = None


# -----------------------------------------------------------------------------


class AsrHermesMqtt:
    """Hermes MQTT server for Rhasspy ASR using Pocketsphinx."""

    def __init__(
        self,
        client,
        transcriber_factory: typing.Callable[[], Transcriber],
        dictionary: Path,
        language_model: Path,
        base_dictionaries: typing.Optional[typing.List[Path]] = None,
        dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        g2p_model: typing.Optional[Path] = None,
        g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        unknown_words: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
        loop=None,
    ):
        self.client = client
        self.make_transcriber = transcriber_factory
        self.transcriber: typing.Optional[Transcriber] = None

        # Files to write during training
        self.dictionary = dictionary
        self.language_model = language_model

        # Pronunciation dictionaries and word transform function
        base_dictionaries = base_dictionaries or []
        self.base_dictionaries = [
            PronunciationDictionary(path=path) for path in base_dictionaries
        ]
        self.dictionary_word_transform = dictionary_word_transform

        # Grapheme-to-phonme model (Phonetisaurus FST) and word transform
        # function.
        self.g2p_model = g2p_model
        self.g2p_word_transform = g2p_word_transform

        # Path to write missing words and guessed pronunciations
        self.unknown_words = unknown_words

        # If True, dictionary and language model won't be overwritten during training
        self.no_overwrite_train = no_overwrite_train

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

        self.first_audio: bool = True

        self.loop = loop or asyncio.get_event_loop()

    # -------------------------------------------------------------------------

    async def start_listening(self, message: AsrStartListening):
        """Start recording audio data for a session."""
        session = self.sessions.get(message.sessionId)
        if not session:
            session = SessionInfo(sessionId=message.sessionId, start_listening=message)

            if message.stopOnSilence:
                # Use voice command recorder
                session.recorder = self.make_recorder()
            else:
                # Use buffer
                session.audio_buffer = bytes()

            self.sessions[message.sessionId] = session

        # Start session
        assert session

        if session.recorder:
            session.recorder.start()

        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    async def stop_listening(
        self, message: AsrStopListening
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Stop recording audio data for a session."""
        try:
            session = self.sessions.pop(message.sessionId, None)
            if session:
                # Stop session
                if session.recorder:
                    audio_data = session.recorder.stop()
                else:
                    assert session.audio_buffer is not None
                    audio_data = session.audio_buffer

                wav_bytes = self.to_wav_bytes(audio_data)

                _LOGGER.debug(
                    "Received a total of %s byte(s) for WAV data for session %s",
                    session.num_wav_bytes,
                    message.sessionId,
                )

                if not session.transcription_sent:
                    # Send transcription
                    session.transcription_sent = True

                    yield (
                        await self.transcribe(
                            wav_bytes,
                            siteId=message.siteId,
                            sessionId=message.sessionId,
                        )
                    )

                    if session.start_listening.sendAudioCaptured:
                        # Send audio data
                        yield (
                            # pylint: disable=E1121
                            AsrAudioCaptured(wav_bytes),
                            {"siteId": message.siteId, "sessionId": message.sessionId},
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

    async def handle_audio_frame(
        self,
        frame_wav_bytes: bytes,
        siteId: str = "default",
        sessionId: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if sessionId is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(sessionId, self.sessions[sessionId])]

        # Add audio to session(s)
        for target_id, session in target_sessions:
            try:
                # Skip if siteId doesn't match
                if session.start_listening.siteId != siteId:
                    continue

                session.num_wav_bytes += len(frame_wav_bytes)
                if session.recorder:
                    # Check for end of voice command
                    command = session.recorder.process_chunk(audio_data)
                    if command and (command.result == VoiceCommandResult.SUCCESS):
                        assert command.audio_data is not None
                        _LOGGER.debug(
                            "Voice command recorded for session %s (%s byte(s))",
                            target_id,
                            len(command.audio_data),
                        )

                        session.transcription_sent = True
                        wav_bytes = self.to_wav_bytes(command.audio_data)

                        yield (
                            await self.transcribe(
                                wav_bytes, siteId=siteId, sessionId=target_id
                            )
                        )

                        if session.start_listening.sendAudioCaptured:
                            # Send audio data
                            yield (
                                AsrAudioCaptured(wav_bytes=wav_bytes),
                                {"siteId": siteId, "sessionId": target_id},
                            )

                        # Reset session (but keep open)
                        session.recorder.stop()
                        session.recorder.start()
                else:
                    # Add to buffer
                    assert session.audio_buffer is not None
                    session.audio_buffer += audio_data
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(self.transcriber),
                    siteId=siteId,
                    sessionId=target_id,
                )

    async def transcribe(
        self, wav_bytes: bytes, siteId: str = "default", sessionId: str = ""
    ) -> typing.Union[AsrTextCaptured, AsrError]:
        """Transcribe audio data and publish captured text."""
        try:
            if not self.transcriber:
                self.transcriber = self.make_transcriber()

            _LOGGER.debug("Transcribing %s byte(s) of audio data", len(wav_bytes))
            transcription = self.transcriber.transcribe_wav(wav_bytes)
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

    async def handle_train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
        _LOGGER.debug("<- %s", train)

        try:
            if not self.base_dictionaries:
                _LOGGER.warning(
                    "No base dictionaries provided. Training will likely fail."
                )

            # Load base dictionaries
            pronunciations: PronunciationsType = defaultdict(list)
            for base_dict in self.base_dictionaries:
                if not os.path.exists(base_dict.path):
                    _LOGGER.warning(
                        "Base dictionary does not exist: %s", base_dict.path
                    )
                    continue

                # Re-load dictionary if modification time has changed
                dict_mtime_ns = os.stat(base_dict.path).st_mtime_ns
                if (base_dict.mtime_ns is None) or (
                    base_dict.mtime_ns != dict_mtime_ns
                ):
                    base_dict.mtime_ns = dict_mtime_ns
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
                        rhasspynlu.g2p.read_pronunciations(
                            base_dict_file, word_dict=base_dict.pronunciations
                        )

                for word in base_dict.pronunciations:
                    pronunciations[word].extend(base_dict.pronunciations[word])

            if not self.no_overwrite_train:
                _LOGGER.debug("Loading %s", train.graph_path)
                with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                    graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

                # Generate dictionary/language model
                _LOGGER.debug("Starting training")
                rhasspyasr_pocketsphinx.train(
                    graph,
                    self.dictionary,
                    self.language_model,
                    pronunciations,
                    dictionary_word_transform=self.dictionary_word_transform,
                    g2p_model=self.g2p_model,
                    g2p_word_transform=self.g2p_word_transform,
                    missing_words_path=self.unknown_words,
                )
            else:
                _LOGGER.warning("Not overwriting dictionary/language model")

            _LOGGER.debug("Re-loading transcriber")
            self.transcriber = self.make_transcriber()

            yield (AsrTrainSuccess(id=train.id), {"siteId": siteId})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=train.id,
            )

    async def handle_pronounce(
        self, pronounce: G2pPronounce
    ) -> typing.AsyncIterable[typing.Union[G2pPhonemes, G2pError]]:
        """Looks up or guesses word pronunciation(s)."""
        try:
            result = G2pPhonemes(
                id=pronounce.id, siteId=pronounce.siteId, sessionId=pronounce.sessionId
            )

            # Load base dictionaries
            pronunciations: typing.Dict[str, typing.List[typing.List[str]]] = {}

            for base_dict in self.base_dictionaries:
                if base_dict.path.is_file():
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
                        rhasspynlu.g2p.read_pronunciations(
                            base_dict_file, word_dict=pronunciations
                        )

            # Try to look up in dictionary first
            missing_words: typing.Set[str] = set()
            if pronunciations:
                for word in pronounce.words:
                    # Handle case transformation
                    if self.dictionary_word_transform:
                        word = self.dictionary_word_transform(word)

                    word_prons = pronunciations.get(word)
                    if word_prons:
                        # Use dictionary pronunciations
                        result.wordPhonemes[word] = [
                            G2pPronunciation(phonemes=p, guessed=False)
                            for p in word_prons
                        ]
                    else:
                        # Will have to guess later
                        missing_words.add(word)
            else:
                # All words must be guessed
                missing_words.update(pronounce.words)

            if missing_words:
                if self.g2p_model:
                    _LOGGER.debug("Guessing pronunciations of %s", missing_words)
                    guesses = rhasspynlu.g2p.guess_pronunciations(
                        missing_words,
                        self.g2p_model,
                        g2p_word_transform=self.g2p_word_transform,
                        num_guesses=pronounce.numGuesses,
                    )

                    # Add guesses to result
                    for guess_word, guess_phonemes in guesses:
                        result_phonemes = result.wordPhonemes.get(guess_word) or []
                        result_phonemes.append(
                            G2pPronunciation(phonemes=guess_phonemes, guessed=True)
                        )
                        result.wordPhonemes[guess_word] = result_phonemes
                else:
                    _LOGGER.warning("No g2p model. Cannot guess pronunciations.")

            yield result
        except Exception as e:
            _LOGGER.exception("handle_pronounce")
            yield G2pError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=pronounce.siteId,
                sessionId=pronounce.id,
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
                G2pPronounce.topic(),
            ]

            if self.siteIds:
                # Specific siteIds
                for siteId in self.siteIds:
                    topics.extend(
                        [
                            AudioFrame.topic(siteId=siteId),
                            AudioSessionFrame.topic(siteId=siteId, sessionId="+"),
                            AsrTrain.topic(siteId=siteId),
                        ]
                    )
            else:
                # All siteIds
                topics.extend(
                    [
                        AudioFrame.topic(siteId="+"),
                        AudioSessionFrame.topic(siteId="+", sessionId="+"),
                        AsrTrain.topic(siteId="+"),
                    ]
                )

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            # Check enable/disable messages
            if msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = True
                    _LOGGER.debug("Enabled")
            elif msg.topic == AsrToggleOff.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = False
                    _LOGGER.debug("Disabled")

            if self.enabled and AudioFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioFrame.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to all active sessions
                    self.publish_all(
                        self.handle_audio_frame(msg.payload, siteId=siteId)
                    )
            elif self.enabled and AudioSessionFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioSessionFrame.get_siteId(msg.topic)
                sessionId = AudioSessionFrame.get_sessionId(msg.topic)
                if ((not self.siteIds) or (siteId in self.siteIds)) and (
                    sessionId in self.sessions
                ):
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    self.publish_all(
                        self.handle_audio_frame(
                            msg.payload, siteId=siteId, sessionId=sessionId
                        )
                    )
            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    asyncio.run_coroutine_threadsafe(
                        self.start_listening(AsrStartListening.from_dict(json_payload)),
                        loop=self.loop,
                    )
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.stop_listening(AsrStopListening.from_dict(json_payload))
                    )
            elif AsrTrain.is_topic(msg.topic):
                # rhasspy/asr/<siteId>/train
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.publish_all(
                        self.handle_train(
                            AsrTrain.from_dict(json_payload), siteId=siteId
                        )
                    )
            elif msg.topic == G2pPronounce.topic():
                # rhasspy/g2p/pronounce
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.handle_pronounce(G2pPronounce.from_dict(json_payload))
                    )
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            if isinstance(message, AsrAudioCaptured):
                _LOGGER.debug(
                    "-> %s(%s byte(s))",
                    message.__class__.__name__,
                    len(message.wav_bytes),
                )
                payload = message.wav_bytes
            else:
                _LOGGER.debug("-> %s", message)
                payload = json.dumps(attr.asdict(message)).encode()

            topic = message.topic(**topic_args)
            _LOGGER.debug("Publishing %s bytes(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("publish")

    def publish_all(self, async_generator: GeneratorType):
        """Publish all messages from an async generator"""
        asyncio.run_coroutine_threadsafe(
            self.async_publish_all(async_generator), self.loop
        )

    async def async_publish_all(self, async_generator: GeneratorType):
        """Enumerate all messages in an async generator publish them"""
        async for maybe_message in async_generator:
            if isinstance(maybe_message, Message):
                self.publish(maybe_message)
            else:
                message, kwargs = maybe_message
                self.publish(message, **kwargs)

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    # -------------------------------------------------------------------------

    def _convert_wav(self, wav_bytes: bytes) -> bytes:
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
            input=wav_bytes,
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

    def to_wav_bytes(self, audio_data: bytes) -> bytes:
        """Wrap raw audio data in WAV."""
        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(self.sample_rate)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setnchannels(self.channels)
                wav_file.writeframes(audio_data)

            return wav_buffer.getvalue()

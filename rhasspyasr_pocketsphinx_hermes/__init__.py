"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import gzip
import logging
import os
import tempfile
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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
    AsrToggleReason,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspyhermes.nlu import AsrToken, AsrTokenTime
from rhasspynlu.g2p import PronunciationsType
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger("rhasspyasr_pocketsphinx_hermes")

# -----------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Information about an open session."""

    start_listening: AsrStartListening
    session_id: typing.Optional[str] = None
    recorder: typing.Optional[VoiceCommandRecorder] = None
    transcription_sent: bool = False
    num_wav_bytes: int = 0
    audio_buffer: typing.Optional[bytes] = None

    # Custom transcriber for filtered intents
    transcriber: typing.Optional[Transcriber] = None


@dataclass
class PronunciationDictionary:
    """Details of a phonetic dictionary."""

    path: Path
    pronunciations: PronunciationsType = field(default_factory=dict)
    mtime_ns: typing.Optional[int] = None


# -----------------------------------------------------------------------------


class AsrHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy ASR using Pocketsphinx."""

    def __init__(
        self,
        client,
        transcriber_factory: typing.Callable[[Path], Transcriber],
        dictionary: Path,
        language_model: Path,
        base_dictionaries: typing.Optional[typing.List[Path]] = None,
        dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        g2p_model: typing.Optional[Path] = None,
        g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        unknown_words: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
        intent_graph_path: typing.Optional[Path] = None,
        base_language_model_fst: typing.Optional[Path] = None,
        base_language_model_weight: float = 0,
        mixed_language_model_fst: typing.Optional[Path] = None,
        site_ids: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
        skip_seconds: float = 0.0,
        min_seconds: float = 1.0,
        speech_seconds: float = 0.3,
        silence_seconds: float = 0.5,
        before_seconds: float = 0.5,
        vad_mode: int = 3,
        lm_cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__(
            "rhasspyasr_pocketsphinx_hermes",
            client,
            site_ids=site_ids,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        self.subscribe(
            AsrToggleOn,
            AsrToggleOff,
            AsrStartListening,
            AsrStopListening,
            G2pPronounce,
            AudioFrame,
            AudioSessionFrame,
            AsrTrain,
        )

        self.make_transcriber = transcriber_factory
        self.transcriber: typing.Optional[Transcriber] = None

        # Intent graph from training
        self.intent_graph_path: typing.Optional[Path] = intent_graph_path
        self.intent_graph: typing.Optional[nx.DiGraph] = None

        # Files to write during training
        self.dictionary = dictionary
        self.language_model = language_model

        # Cache for filtered language model
        self.lm_cache_dir = lm_cache_dir
        self.lm_cache_paths: typing.Dict[str, Path] = {}
        self.lm_cache_transcribers: typing.Dict[str, Transcriber] = {}

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

        # Mixed language model
        self.base_language_model_fst = base_language_model_fst
        self.base_language_model_weight = base_language_model_weight
        self.mixed_language_model_fst = mixed_language_model_fst

        # If True, dictionary and language model won't be overwritten during training
        self.no_overwrite_train = no_overwrite_train

        # True if ASR system is enabled
        self.enabled = enabled
        self.disabled_reasons: typing.Set[str] = set()

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(
                max_seconds=None,
                vad_mode=vad_mode,
                skip_seconds=skip_seconds,
                min_seconds=min_seconds,
                speech_seconds=speech_seconds,
                silence_seconds=silence_seconds,
                before_seconds=before_seconds,
            )

        self.make_recorder = make_recorder or default_recorder

        # WAV buffers for each session
        self.sessions: typing.Dict[typing.Optional[str], SessionInfo] = {}

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    async def start_listening(self, message: AsrStartListening) -> None:
        """Start recording audio data for a session."""
        session = self.sessions.get(message.session_id)
        if not session:
            session = SessionInfo(
                session_id=message.session_id, start_listening=message
            )

            if message.stop_on_silence:
                # Use voice command recorder
                session.recorder = self.make_recorder()
            else:
                # Use buffer
                session.audio_buffer = bytes()

            if message.intent_filter:
                # Load filtered language model
                self.maybe_load_filtered_transcriber(session, message.intent_filter)

            self.sessions[message.session_id] = session

        # Start session
        assert session

        if session.recorder:
            session.recorder.start()

        _LOGGER.debug("Starting listening (session_id=%s)", message.session_id)
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
            session = self.sessions.pop(message.session_id, None)
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
                    message.session_id,
                )

                if not session.transcription_sent:
                    # Send transcription
                    session.transcription_sent = True

                    yield (
                        await self.transcribe(
                            wav_bytes,
                            transcriber=session.transcriber,
                            site_id=message.site_id,
                            session_id=message.session_id,
                            lang=session.start_listening.lang,
                        )
                    )

                    if session.start_listening.send_audio_captured:
                        # Send audio data
                        yield (
                            AsrAudioCaptured(wav_bytes=wav_bytes),
                            {
                                "site_id": message.site_id,
                                "session_id": message.session_id,
                            },
                        )

            _LOGGER.debug("Stopping listening (session_id=%s)", message.session_id)
        except Exception as e:
            _LOGGER.exception("stop_listening")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                site_id=message.site_id,
                session_id=message.session_id,
            )

    async def handle_audio_frame(
        self,
        frame_wav_bytes: bytes,
        site_id: str = "default",
        session_id: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Process single frame of WAV audio"""
        # Don't process audio if no sessions
        if not self.sessions:
            return

        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if session_id is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(session_id, self.sessions[session_id])]

        # Add audio to session(s)
        for target_id, session in target_sessions:
            try:
                # Skip if site_id doesn't match
                if session.start_listening.site_id != site_id:
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
                                wav_bytes,
                                transcriber=session.transcriber,
                                site_id=site_id,
                                session_id=target_id,
                                lang=session.start_listening.lang,
                            )
                        )

                        if session.start_listening.send_audio_captured:
                            # Send audio data
                            yield (
                                AsrAudioCaptured(wav_bytes=wav_bytes),
                                {"site_id": site_id, "session_id": target_id},
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
                    site_id=site_id,
                    session_id=target_id,
                )

    async def transcribe(
        self,
        wav_bytes: bytes,
        site_id: str,
        transcriber: typing.Optional[Transcriber] = None,
        session_id: typing.Optional[str] = None,
        lang: typing.Optional[str] = None,
    ) -> AsrTextCaptured:
        """Transcribe audio data and publish captured text."""
        if not transcriber and not self.transcriber:
            # Load default transcriber
            self.transcriber = self.make_transcriber(self.language_model)

        transcriber = transcriber or self.transcriber
        assert transcriber, "No transcriber"

        _LOGGER.debug("Transcribing %s byte(s) of audio data", len(wav_bytes))
        transcription = transcriber.transcribe_wav(wav_bytes)
        if transcription:
            _LOGGER.debug(transcription)
            asr_tokens: typing.Optional[typing.List[typing.List[AsrToken]]] = None

            if transcription.tokens:
                # Only one level of ASR tokens
                asr_inner_tokens: typing.List[AsrToken] = []
                asr_tokens = [asr_inner_tokens]
                range_start = 0
                for ps_token in transcription.tokens:
                    range_end = range_start + len(ps_token.token) + 1
                    asr_inner_tokens.append(
                        AsrToken(
                            value=ps_token.token,
                            confidence=ps_token.likelihood,
                            range_start=range_start,
                            range_end=range_start + len(ps_token.token) + 1,
                            time=AsrTokenTime(
                                start=ps_token.start_time, end=ps_token.end_time
                            ),
                        )
                    )

                    range_start = range_end

            # Actual transcription
            return AsrTextCaptured(
                text=transcription.text,
                likelihood=transcription.likelihood,
                seconds=transcription.transcribe_seconds,
                site_id=site_id,
                session_id=session_id,
                asr_tokens=asr_tokens,
                lang=lang,
            )

        _LOGGER.warning("Received empty transcription")
        return AsrTextCaptured(
            text="",
            likelihood=0,
            seconds=0,
            site_id=site_id,
            session_id=session_id,
            lang=lang,
        )

    async def handle_train(
        self, train: AsrTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
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

            # Load intent graph
            _LOGGER.debug("Loading %s", train.graph_path)
            with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                self.intent_graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

            # Clean LM cache completely
            for lm_path in self.lm_cache_paths.values():
                try:
                    lm_path.unlink()
                except Exception:
                    pass

            self.lm_cache_paths = {}
            self.lm_cache_transcribers = {}

            # Generate dictionary/language model
            if not self.no_overwrite_train:
                _LOGGER.debug("Starting training")
                rhasspyasr_pocketsphinx.train(
                    self.intent_graph,
                    self.dictionary,
                    self.language_model,
                    pronunciations,
                    dictionary_word_transform=self.dictionary_word_transform,
                    g2p_model=self.g2p_model,
                    g2p_word_transform=self.g2p_word_transform,
                    missing_words_path=self.unknown_words,
                    base_language_model_fst=self.base_language_model_fst,
                    base_language_model_weight=self.base_language_model_weight,
                    mixed_language_model_fst=self.mixed_language_model_fst,
                )
            else:
                _LOGGER.warning("Not overwriting dictionary/language model")

            _LOGGER.debug("Re-loading transcriber")
            self.transcriber = self.make_transcriber(self.language_model)

            yield (AsrTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                site_id=site_id,
                session_id=train.id,
            )

    async def handle_pronounce(
        self, pronounce: G2pPronounce
    ) -> typing.AsyncIterable[typing.Union[G2pPhonemes, G2pError]]:
        """Looks up or guesses word pronunciation(s)."""
        try:
            result = G2pPhonemes(
                word_phonemes={},
                id=pronounce.id,
                site_id=pronounce.site_id,
                session_id=pronounce.session_id,
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
                        result.word_phonemes[word] = [
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
                        num_guesses=pronounce.num_guesses,
                    )

                    # Add guesses to result
                    for guess_word, guess_phonemes in guesses:
                        result_phonemes = result.word_phonemes.get(guess_word) or []
                        result_phonemes.append(
                            G2pPronunciation(phonemes=guess_phonemes, guessed=True)
                        )
                        result.word_phonemes[guess_word] = result_phonemes
                else:
                    _LOGGER.warning("No g2p model. Cannot guess pronunciations.")

            yield result
        except Exception as e:
            _LOGGER.exception("handle_pronounce")
            yield G2pError(
                error=str(e),
                context=repr(self.transcriber),
                site_id=pronounce.site_id,
                session_id=pronounce.session_id,
            )

    def cleanup(self):
        """Delete any temporary files."""
        for lm_path in self.lm_cache_paths.values():
            try:
                lm_path.unlink()
            except Exception:
                pass

    def maybe_load_filtered_transcriber(
        self, session: SessionInfo, intent_filter: typing.List[str]
    ):
        """Create/load a language model with only filtered intents."""
        lm_key = ",".join(intent_filter)

        # Try to look up in cache
        lm_transcriber = self.lm_cache_transcribers.get(lm_key)

        if not lm_transcriber:
            lm_path = self.lm_cache_paths.get(lm_key)

            if not lm_path:
                # Create a new temporary file
                lm_file = tempfile.NamedTemporaryFile(
                    suffix=".arpa", dir=self.lm_cache_dir, delete=False
                )
                lm_path = Path(lm_file.name)
                self.lm_cache_paths[lm_key] = lm_path

            # Function to filter intents by name
            def intent_filter_func(intent_name: str) -> bool:
                return intent_name in intent_filter

            # Load intent graph and create transcriber
            if (
                not self.intent_graph
                and self.intent_graph_path
                and self.intent_graph_path.is_file()
            ):
                # Load intent graph
                _LOGGER.debug("Loading %s", self.intent_graph_path)
                with gzip.GzipFile(self.intent_graph_path, mode="rb") as graph_gzip:
                    self.intent_graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

            if self.intent_graph:
                # Create language model
                _LOGGER.debug("Converting to ARPA language model")
                rhasspynlu.arpa_lm.graph_to_arpa(
                    self.intent_graph, lm_path, intent_filter=intent_filter_func
                )

                # Load transcriber
                lm_transcriber = self.make_transcriber(lm_path)
                self.lm_cache_transcribers[lm_key] = lm_transcriber
            else:
                # Use full transcriber
                _LOGGER.warning("No intent graph loaded. Cannot filter intents.")

        session.transcriber = lm_transcriber

    # -------------------------------------------------------------------------

    async def on_message_blocking(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker (blocking)."""
        # Check enable/disable messages
        if isinstance(message, AsrToggleOn):
            if message.reason == AsrToggleReason.UNKNOWN:
                # Always enable on unknown
                self.disabled_reasons.clear()
            else:
                self.disabled_reasons.discard(message.reason)

            if self.disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.disabled_reasons)
            else:
                self.enabled = True
                self.first_audio = True
                _LOGGER.debug("Enabled")
        elif isinstance(message, AsrToggleOff):
            self.enabled = False
            self.disabled_reasons.add(message.reason)
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert site_id, "Missing site_id"
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Add to all active sessions
                async for frame_result in self.handle_audio_frame(
                    message.wav_bytes, site_id=site_id
                ):
                    yield frame_result
        elif isinstance(message, AudioSessionFrame):
            if self.enabled:
                assert site_id and session_id, "Missing site_id or session_id"
                if session_id in self.sessions:
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    async for session_frame_result in self.handle_audio_frame(
                        message.wav_bytes, site_id=site_id, session_id=session_id
                    ):
                        yield session_frame_result
        elif isinstance(message, AsrStartListening):
            # Handle blocking
            await self.start_listening(message)
        elif isinstance(message, AsrStopListening):
            # hermes/asr/stopListening
            async for stop_result in self.stop_listening(message):
                yield stop_result
        elif isinstance(message, AsrTrain):
            # rhasspy/asr/<site_id>/train
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        elif isinstance(message, G2pPronounce):
            # rhasspy/g2p/pronounce
            async for pronounce_result in self.handle_pronounce(message):
                yield pronounce_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)

"""Provisional messages for hermes/asr"""
import re
import typing

import attr

from rhasspyhermes.base import Message


@attr.s
class AsrError(Message):
    """Error from ASR component."""

    error: str = attr.ib()
    context: str = attr.ib(default="")
    siteId: str = attr.ib(default="default")
    sessionId: str = attr.ib(default="")

    @classmethod
    def topic(cls, **kwargs) -> str:
        """Get Hermes topic"""
        return "hermes/error/asr"


@attr.s
class AsrTrain(Message):
    """Request to retrain from intent graph"""

    TOPIC_PATTERN = re.compile(r"^hermes/asr/([^/]+)/train$")

    id: str = attr.ib()
    graph_dict: typing.Dict[str, typing.Any] = attr.ib()

    @classmethod
    def topic(cls, **kwargs) -> str:
        siteId = kwargs.get("siteId", "default")
        return f"hermes/asr/{siteId}/train"

    @classmethod
    def is_topic(cls, topic: str) -> bool:
        """True if topic matches template"""
        return re.match(AsrTrain.TOPIC_PATTERN, topic) is not None

    @classmethod
    def get_siteId(cls, topic: str) -> str:
        """Get siteId from a topic"""
        match = re.match(AsrTrain.TOPIC_PATTERN, topic)
        assert match, "Not a train topic"
        return match.group(1)


@attr.s
class AsrTrainSuccess(Message):
    """Result from successful training"""

    TOPIC_PATTERN = re.compile(r"^hermes/asr/([^/]+)/trainSuccess$")

    id: str = attr.ib()

    @classmethod
    def topic(cls, **kwargs) -> str:
        siteId = kwargs.get("siteId", "default")
        return f"hermes/asr/{siteId}/trainSuccess"

    @classmethod
    def is_topic(cls, topic: str) -> bool:
        """True if topic matches template"""
        return re.match(AsrTrainSuccess.TOPIC_PATTERN, topic) is not None

    @classmethod
    def get_siteId(cls, topic: str) -> str:
        """Get siteId from a topic"""
        match = re.match(AsrTrainSuccess.TOPIC_PATTERN, topic)
        assert match, "Not a trainSuccess topic"
        return match.group(1)

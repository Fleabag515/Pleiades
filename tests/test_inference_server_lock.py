"""EngineState.chat() streaming must hold self.lock for the generator's
whole lifetime, not just until chat() returns -- otherwise a concurrent
resize()/second request can run against the same llama context mid-stream.
See HANDOFF's 2026-07-25 correctness audit, finding #2."""

import threading

from pleiades.inference.server import EngineState


class _FakeLlama:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def create_chat_completion(self, stream=False, **kw):
        assert stream is True
        return iter(self._chunks)


def _bare_state(chunks) -> EngineState:
    # Bypass __init__ (it loads a real GGUF) and wire up just what chat() touches.
    st = object.__new__(EngineState)
    st.lock = threading.Lock()
    st.llama = _FakeLlama(chunks)
    st.metrics = {"requests": 0, "streamed": 0, "resizes": 0, "overflow_upshifts": 0}
    return st


def test_stream_holds_lock_until_fully_consumed():
    st = _bare_state([{"c": 1}, {"c": 2}, {"c": 3}])
    payload, stream = st.chat({"messages": [], "stream": True})
    assert payload is None
    assert st.lock.locked(), "lock must still be held after chat() returns a stream"

    collected = list(stream)  # drain it

    assert collected == [{"c": 1}, {"c": 2}, {"c": 3}]
    assert not st.lock.locked(), "lock must be released once the stream is exhausted"


def test_stream_releases_lock_on_early_close():
    st = _bare_state([{"c": 1}, {"c": 2}, {"c": 3}])
    _, stream = st.chat({"messages": [], "stream": True})
    next(stream)  # consume one chunk, then abandon it
    assert st.lock.locked()

    stream.close()

    assert not st.lock.locked(), "closing the generator early must still release the lock"


def test_non_stream_does_not_hold_lock():
    st = _bare_state([])
    st.llama.create_chat_completion = lambda stream=False, **kw: {"ok": True}
    payload, stream = st.chat({"messages": [], "stream": False})
    assert payload == {"ok": True}
    assert stream is None
    assert not st.lock.locked()


if __name__ == "__main__":
    test_stream_holds_lock_until_fully_consumed()
    test_stream_releases_lock_on_early_close()
    test_non_stream_does_not_hold_lock()
    print("ok")

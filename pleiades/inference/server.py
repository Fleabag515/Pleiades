"""Pleiades' elastic inference server.

A thin OpenAI-compatible HTTP server over `llama_cpp.Llama` that Pleiades
fully owns — which is what makes *elastic context* possible:

  - weights are loaded once per process (mmap'd); a resize rebuilds only the
    llama context at a new `n_ctx`, so growing/shrinking the KV cache costs
    seconds, not a model reload from disk
  - `POST /resize {"n_ctx": N}` — the upshift endpoint Anamnesis' governor
    calls under context pressure (clamped to the planner's ceiling)
  - prompt overflow self-heals: if a request exceeds the current window and a
    bigger gear is allowed, the server upshifts and retries once instead of
    erroring
  - `GET /props` advertises {n_ctx, n_ctx_train, n_ctx_max, resizable:true}
    (llama-server-compatible shape) so proxies can *discover* the window
    instead of being configured with it
  - `POST /tokenize` + `POST /extras/tokenize/count` give proxies real token
    counts (llama-server and llama-cpp-python shapes respectively)

Endpoints:
  GET  /            health
  GET  /v1/models   OpenAI model list
  POST /v1/chat/completions   (stream + non-stream)
  GET  /props       window + elasticity facts
  GET  /metrics     counters
  POST /tokenize    {content} -> {tokens:[...]}
  POST /extras/tokenize/count {input} -> {count}
  POST /resize      {n_ctx} -> {n_ctx, took_ms}

stdlib only (ThreadingHTTPServer); llama.cpp is single-stream for one model,
so requests serialize on a lock — exactly the behaviour of one llama-server
slot. Run via:  python -m pleiades.inference.server --model x.gguf ...
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..hardware import (CONTEXT_GEARS, kv_bytes_per_token, read_gguf_meta,
                        resolve_layers)


class EngineState:
    """The loaded model + its current context, swappable under a lock."""

    def __init__(self, model_path: str, n_ctx: int, n_ctx_max: int,
                 n_gpu_layers: "int | str", chat_format: str, alias: str):
        self.model_path = model_path
        self.alias = alias or "local"
        self.n_ctx = n_ctx
        self.n_ctx_max = n_ctx_max
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.meta = read_gguf_meta(model_path)
        self.lock = threading.Lock()  # serializes generation AND resize
        self.llama = None
        self.metrics = {"requests": 0, "streamed": 0, "resizes": 0,
                        "overflow_upshifts": 0, "started_at": time.time()}

    # -- lifecycle ---------------------------------------------------------- #
    def load(self, n_ctx: Optional[int] = None) -> None:
        """(Re)create the llama context. Call with the lock held (or at boot)."""
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "llama-cpp-python is not installed. "
                "pip install 'llama-cpp-python'"
            ) from e

        if n_ctx is not None:
            self.n_ctx = n_ctx
        layers, why = resolve_layers(self.n_gpu_layers, self.model_path, self.n_ctx)
        print(f"[engine] loading n_ctx={self.n_ctx} n_gpu_layers={layers} — {why}",
              flush=True)

        if self.llama is not None:
            # Free the old context/weights handle first; mmap keeps the file
            # hot in page cache, so the reload below is fast.
            try:
                self.llama.close()
            except AttributeError:
                pass
            self.llama = None

        kwargs = dict(model_path=self.model_path, n_ctx=self.n_ctx,
                      n_gpu_layers=layers, verbose=False)
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        try:
            self.llama = Llama(flash_attn=True, **kwargs)
        except TypeError:  # older llama-cpp-python without flash_attn kwarg
            self.llama = Llama(**kwargs)

    # -- elastic ------------------------------------------------------------ #
    def clamp_gear(self, requested: int) -> int:
        ceiling = self.n_ctx_max or (self.meta.n_ctx_train or CONTEXT_GEARS[-1])
        return max(1024, min(int(requested), ceiling))

    def next_gear(self) -> Optional[int]:
        for g in CONTEXT_GEARS:
            if g > self.n_ctx and g <= (self.n_ctx_max or g):
                return g
        return None

    def resize(self, requested: int) -> dict:
        target = self.clamp_gear(requested)
        with self.lock:
            if target == self.n_ctx:
                return {"n_ctx": self.n_ctx, "took_ms": 0, "note": "already there"}
            t0 = time.time()
            self.load(target)
            self.metrics["resizes"] += 1
            return {"n_ctx": self.n_ctx, "took_ms": int((time.time() - t0) * 1000)}

    # -- inference ---------------------------------------------------------- #
    _OVERFLOW_MARKERS = ("exceed context window", "context window", "n_ctx")

    def chat(self, body: dict):
        """Run a chat completion; auto-upshift once on prompt overflow.

        Returns (payload, stream_iter): exactly one is non-None. For a
        streaming response the lock is held for the generator's entire
        lifetime, not just until this method returns -- llama.cpp is
        single-stream per context, so a second caller (or a concurrent
        resize()/load()) must block until this stream is fully consumed or
        closed, not just until chat() hands the generator back.
        """
        allowed = {"messages", "tools", "tool_choice", "temperature", "top_p",
                   "top_k", "min_p", "max_tokens", "stream", "stop", "seed",
                   "presence_penalty", "frequency_penalty", "repeat_penalty",
                   "response_format", "logit_bias"}
        kw = {k: v for k, v in body.items() if k in allowed and v is not None}
        stream = bool(kw.pop("stream", False))
        self.metrics["requests"] += 1

        for attempt in (0, 1):
            self.lock.acquire()
            commit_stream = False
            try:
                if not stream:
                    return self.llama.create_chat_completion(stream=False, **kw), None
                it = self.llama.create_chat_completion(stream=True, **kw)
                first = next(it, None)  # prompt eval happens here
                self.metrics["streamed"] += 1
                commit_stream = True
            except ValueError as e:
                msg = str(e).lower()
                overflow = any(m in msg for m in self._OVERFLOW_MARKERS)
                nxt = self.next_gear()
                if not (overflow and attempt == 0 and nxt):
                    raise
                print(f"[engine] prompt overflow at n_ctx={self.n_ctx} — "
                      f"upshifting to {nxt} and retrying", flush=True)
                self.load(nxt)
                self.metrics["overflow_upshifts"] += 1
                continue
            finally:
                if not commit_stream:
                    self.lock.release()

            # Committed to streaming: the generator now owns the lock and
            # releases it in `finally` -- on normal completion, on the
            # caller closing/abandoning the generator early (GeneratorExit),
            # or on an error from `rest`. Never released back here.
            def gen(first_chunk=first, rest=it):
                try:
                    if first_chunk is not None:
                        yield first_chunk
                    yield from rest
                finally:
                    self.lock.release()

            return None, gen()
        raise RuntimeError("unreachable")

    def props(self) -> dict:
        return {
            "n_ctx": self.n_ctx,
            "n_ctx_train": self.meta.n_ctx_train,
            "n_ctx_max": self.n_ctx_max,
            "resizable": True,
            "model_path": self.model_path,
            "kv_bytes_per_token": kv_bytes_per_token(self.meta),
            "gears": [g for g in CONTEXT_GEARS if g <= (self.n_ctx_max or g)],
            # llama-server compatible shape for generic discoverers:
            "default_generation_settings": {"n_ctx": self.n_ctx},
        }


def make_handler(state: EngineState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # quiet; the engine prints what matters
            pass

        # -- plumbing ------------------------------------------------------- #
        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                return {}

        # -- GET ------------------------------------------------------------ #
        def do_GET(self):
            if self.path in ("/", "/health"):
                return self._json(200, {"status": "ok", "model": state.alias})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": state.alias, "object": "model",
                     "owned_by": "pleiades", "created": 0}]})
            if self.path == "/props":
                return self._json(200, state.props())
            if self.path == "/metrics":
                return self._json(200, {**state.metrics, "n_ctx": state.n_ctx})
            self._json(404, {"error": "not found"})

        # -- POST ----------------------------------------------------------- #
        def do_POST(self):
            body = self._body()
            if self.path == "/resize":
                n = body.get("n_ctx")
                if not isinstance(n, int) or n <= 0:
                    return self._json(400, {"error": "n_ctx (positive int) required"})
                try:
                    return self._json(200, state.resize(n))
                except Exception as e:  # surface load failures honestly
                    return self._json(500, {"error": f"resize failed: {e}"})
            if self.path == "/tokenize":
                toks = self._tokens(str(body.get("content", "")))
                return self._json(200, {"tokens": toks})
            if self.path == "/extras/tokenize/count":
                toks = self._tokens(str(body.get("input", "")))
                return self._json(200, {"count": len(toks)})
            if self.path.endswith("/chat/completions"):
                return self._chat(body)
            self._json(404, {"error": "not found"})

        def _tokens(self, text: str) -> list:
            with state.lock:
                return state.llama.tokenize(text.encode("utf-8"), add_bos=False,
                                            special=True)

        def _chat(self, body: dict) -> None:
            try:
                payload, stream = state.chat(body)
            except ValueError as e:
                return self._json(400, {"error": {"message": str(e),
                                                  "type": "invalid_request_error"}})
            except Exception as e:
                return self._json(500, {"error": {"message": str(e),
                                                  "type": "server_error"}})
            if payload is not None:
                return self._json(200, payload)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for chunk in stream:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client went away mid-stream

    return Handler


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Pleiades elastic inference server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-ctx", "--n_ctx", dest="n_ctx", default="auto",
                    help="'auto' (planned from VRAM) or an integer")
    ap.add_argument("--n-ctx-max", "--n_ctx_max", dest="n_ctx_max", type=int, default=0,
                    help="elastic ceiling; 0 = planner/trained-context decides")
    ap.add_argument("--n-gpu-layers", "--n_gpu_layers", dest="n_gpu_layers", default="auto")
    ap.add_argument("--chat-format", "--chat_format", dest="chat_format", default="")
    ap.add_argument("--alias", default="local")
    args = ap.parse_args(argv)

    from ..hardware import resolve_context
    n_ctx, planned_max, why = resolve_context(args.n_ctx, args.model)
    n_ctx_max = args.n_ctx_max or planned_max
    print(f"[engine] context plan: {why}", flush=True)

    state = EngineState(args.model, n_ctx, n_ctx_max, args.n_gpu_layers,
                        args.chat_format, args.alias)
    state.load()

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[engine] elastic server on {args.host}:{args.port} "
          f"(n_ctx {n_ctx}, ceiling {n_ctx_max})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

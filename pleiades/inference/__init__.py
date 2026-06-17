"""Pleiades' own inference engine.

This is the part we own: Pleiades loads a GGUF model with llama-cpp-python and serves
an OpenAI-compatible endpoint *in-process* (no external Ollama / server to install).
Anamnesis's `upstream.baseUrl` is pointed at this endpoint, so the data path is:

    engine -> Anamnesis character proxy -> THIS inference server (GGUF via llama.cpp)

Any GGUF that llama.cpp supports works, including large quantized models with GPU
offload (build llama-cpp-python with the appropriate CMAKE_ARGS; see README).

We launch llama_cpp's bundled OpenAI-compatible server as a child process. That keeps
the heavy model in its own process (clean shutdown, no GIL contention with the agent
loop) while remaining entirely managed by Pleiades. llama.cpp is the swappable compute
kernel; the engine layer is ours.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

import httpx

from ..config import Settings


class InferenceError(RuntimeError):
    pass


class InferenceServer:
    """Manages an in-process llama.cpp OpenAI-compatible server over a GGUF model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._proc: Optional[subprocess.Popen] = None

    # -- properties --------------------------------------------------------- #
    @property
    def host(self) -> str:
        return self.settings.inference_host

    @property
    def port(self) -> int:
        return self.settings.inference_port

    @property
    def base_url(self) -> str:
        return self.settings.inference_base_url  # http://host:port/v1

    @property
    def _models_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"

    # -- health ------------------------------------------------------------- #
    def is_running(self) -> bool:
        """True if something is already serving the OpenAI API at host:port."""
        try:
            r = httpx.get(self._models_url, timeout=1.5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise InferenceError(
                    f"Inference server exited early (code {self._proc.returncode}). "
                    "Check the model path and that llama-cpp-python is installed."
                )
            time.sleep(0.5)
        raise InferenceError(f"Inference server not ready after {timeout:.0f}s.")

    # -- lifecycle ---------------------------------------------------------- #
    def _build_cmd(self) -> list[str]:
        if not self.settings.model_path:
            raise InferenceError(
                "PLEIADES_MODEL_PATH is not set. Point it at a .gguf model file."
            )
        import os

        from ..hardware import resolve_context, resolve_layers

        n_ctx, n_ctx_max, ctx_why = resolve_context(self.settings.n_ctx,
                                                    self.settings.model_path)
        layers, why = resolve_layers(self.settings.n_gpu_layers,
                                     self.settings.model_path, n_ctx)
        print(f"[pleiades] inference: context — {ctx_why}")
        print(f"[pleiades] inference: n_gpu_layers={layers} — {why}")
        if os.environ.get("PLEIADES_ENGINE", "elastic").lower() == "llama_cpp":
            # escape hatch: bundled llama-cpp-python server (fixed window)
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", self.settings.model_path,
                "--host", self.host, "--port", str(self.port),
                "--n_ctx", str(n_ctx),
                "--n_gpu_layers", str(layers),
            ]
            if self.settings.chat_format:
                cmd += ["--chat_format", self.settings.chat_format]
            return cmd
        # our elastic server: weights stay loaded, KV resizes in place
        cmd = [
            sys.executable, "-m", "pleiades.inference.server",
            "--model", self.settings.model_path,
            "--host", self.host, "--port", str(self.port),
            "--n-ctx", str(n_ctx), "--n-ctx-max", str(n_ctx_max),
            "--n-gpu-layers", str(layers),
        ]
        if self.settings.chat_format:
            cmd += ["--chat-format", self.settings.chat_format]
        return cmd

    def start(self, *, wait: bool = True, timeout: float = 120.0) -> str:
        """Launch the server (idempotent). Returns the OpenAI-compatible base URL."""
        if self.is_running():
            return self.base_url
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as e:
            raise InferenceError(
                "Could not launch llama_cpp.server. Install with "
                "`pip install 'llama-cpp-python[server]'`."
            ) from e
        if wait:
            self.wait_ready(timeout=timeout)
        return self.base_url

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> "InferenceServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def ensure_inference(settings: Settings, *, wait: bool = True) -> str:
    """Ensure an inference server is reachable and return its OpenAI-compatible URL.

    If the user overrode the backend (PLEIADES_BACKEND_BASE_URL), trust that and do not
    launch anything. Otherwise start our llama.cpp server if it isn't already running.
    """
    if settings.backend_base_url:
        return settings.backend_base_url
    server = InferenceServer(settings)
    if server.is_running():
        return server.base_url
    return server.start(wait=wait)

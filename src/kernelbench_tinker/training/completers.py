from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tinker
from tinker_cookbook.completers import (
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass
class MaskedTokensWithLogprobs(TokensWithLogprobs):
    maybe_mask: list[float] | None = None

    @property
    def mask(self) -> list[float]:
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


@dataclass
class TwoPhaseTokenCompleter(TokenCompleter):
    sampling_client: tinker.SamplingClient
    tokenizer: TokenizerLike
    phase1_max_tokens: int
    temperature: float = 1.0
    context_window: int = 32768
    context_buffer: int = 50

    PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def _hit_stop_sequence(self, tokens: list[int], stop: StopCondition) -> bool:
        if not tokens:
            return False
        for s in stop:
            if isinstance(s, int):
                if tokens[-1] == s:
                    return True
            else:
                stop_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                if len(stop_tokens) <= len(tokens) and tokens[-len(stop_tokens):] == stop_tokens:
                    return True
        return False

    def _contains_subsequence(self, tokens: list[int], pattern: str) -> bool:
        pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_tokens) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_tokens) + 1):
            if tokens[i : i + len(pattern_tokens)] == pattern_tokens:
                return True
        return False

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        prompt_length = model_input.length
        phase1_max = self.phase1_max_tokens - prompt_length
        if phase1_max <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} exceeds phase1_max_tokens {self.phase1_max_tokens}."
            )

        phase1_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=phase1_max,
                temperature=self.temperature,
            ),
        )
        phase1_sequence = phase1_result.sequences[0]
        phase1_tokens = phase1_sequence.tokens
        phase1_logprobs = phase1_sequence.logprobs
        assert phase1_logprobs is not None

        if self._hit_stop_sequence(phase1_tokens, stop) or len(phase1_tokens) < phase1_max:
            return MaskedTokensWithLogprobs(
                tokens=phase1_tokens,
                maybe_logprobs=phase1_logprobs,
                stop_reason=phase1_sequence.stop_reason,
            )

        if self._contains_subsequence(phase1_tokens, self.GPTOSS_FINAL_CHANNEL_INDICATOR):
            new_chunks = list(model_input.chunks) + [
                tinker.types.EncodedTextChunk(tokens=phase1_tokens)
            ]
            phase2_max = (
                self.context_window - prompt_length - len(phase1_tokens) - self.context_buffer
            )
            if phase2_max <= 0:
                return MaskedTokensWithLogprobs(
                    tokens=phase1_tokens,
                    maybe_logprobs=phase1_logprobs,
                    stop_reason="length",
                )
            phase2_result = await self.sampling_client.sample_async(
                prompt=tinker.ModelInput(chunks=new_chunks),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    stop=stop,
                    max_tokens=phase2_max,
                    temperature=self.temperature,
                ),
            )
            phase2_sequence = phase2_result.sequences[0]
            phase2_tokens = phase2_sequence.tokens
            phase2_logprobs = phase2_sequence.logprobs
            assert phase2_logprobs is not None
            return MaskedTokensWithLogprobs(
                tokens=phase1_tokens + phase2_tokens,
                maybe_logprobs=phase1_logprobs + phase2_logprobs,
                stop_reason=phase2_sequence.stop_reason,
            )

        end_token_seq = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        ends_with_end = (
            len(end_token_seq) <= len(phase1_tokens)
            and phase1_tokens[-len(end_token_seq) :] == end_token_seq
        )
        if ends_with_end:
            prefill_text = self.PHASE2_PREFILL + "<|start|>assistant<|channel|>final<|message|>"
        else:
            prefill_text = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
        prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)

        new_chunks = list(model_input.chunks) + [
            tinker.types.EncodedTextChunk(tokens=phase1_tokens),
            tinker.types.EncodedTextChunk(tokens=prefill_tokens),
        ]
        phase2_max = (
            self.context_window
            - prompt_length
            - len(phase1_tokens)
            - len(prefill_tokens)
            - self.context_buffer
        )
        if phase2_max <= 0:
            return MaskedTokensWithLogprobs(
                tokens=phase1_tokens + prefill_tokens,
                maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens),
                maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens),
                stop_reason="length",
            )

        phase2_result = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput(chunks=new_chunks),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=phase2_max,
                temperature=self.temperature,
            ),
        )
        phase2_sequence = phase2_result.sequences[0]
        phase2_tokens = phase2_sequence.tokens
        phase2_logprobs = phase2_sequence.logprobs
        assert phase2_logprobs is not None

        return MaskedTokensWithLogprobs(
            tokens=phase1_tokens + prefill_tokens + phase2_tokens,
            maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens) + phase2_logprobs,
            maybe_mask=[1.0] * len(phase1_tokens)
            + [0.0] * len(prefill_tokens)
            + [1.0] * len(phase2_tokens),
            stop_reason=phase2_sequence.stop_reason,
        )


def build_token_completer(
    sampling_client: tinker.SamplingClient,
    max_tokens: int,
    temperature: float,
    use_two_phase_sampling: bool,
    tokenizer: TokenizerLike | None = None,
    phase1_max_tokens: int = 26000,
    context_window: int = 32768,
    context_buffer: int = 50,
) -> TokenCompleter:
    if use_two_phase_sampling:
        if tokenizer is None:
            raise ValueError("tokenizer is required when use_two_phase_sampling is enabled")
        return TwoPhaseTokenCompleter(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            phase1_max_tokens=phase1_max_tokens,
            temperature=temperature,
            context_window=context_window,
            context_buffer=context_buffer,
        )
    return TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )

""" TextSynth API
Implementation provided by Harsh Raj

In order to use the API, you must have a valid Replicate account and
enough credits.

Example usage:

    python main.py --model replicate --model_args name=gptj_6B --no_cache --tasks truthfulqa

Homepage: https://replicate.com
"""
import time
import logging
import os
from typing import Any, List, Tuple

import requests as _requests
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions

from replicate.__about__ import __version__

logger = logging.getLogger(__name__)


# def textsynth_completion(**kwargs):
#     """Query TextSynth API for completion.
#     Retry with back-off until they respond.
#     """

#     def _exception_callback(e: Exception, sleep_time: float) -> None:
#         import traceback

#         traceback.print_exc()

#     @retry_on_specific_exceptions(
#         on_exceptions=[_requests.exceptions.RequestException],
#         max_retries=None,  # retry forever, consider changing
#         on_exception_callback=_exception_callback,
#     )
#     def completion():
#         return _requests.post(**kwargs)

#     return completion()


@register_model("replicate")
class ReplicateLM(LM):
    def __init__(
        self, 
        name, 
        temperature: float = 0.1, 
        **kwargs, # top_p, top_k, etc.
        ) -> None:
        """
        :param name: str
            Replicate API name (e.g. `gptj_6B`)
        :param temperature: float
            Sampling temperature
        """
        super().__init__()

        _, self.version = name.split(':')
        self.temperature = temperature
        self.api_url = "https://api.replicate.com"
        # Read from environment variable REPLICATE_API_TOKEN
        self.api_key = os.environ["REPLICATE_API_TOKEN"]

    @property
    def eot_token_id(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    @property
    def max_length(self) -> int:
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def tok_decode(self, tokens):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests):
            response = textsynth_completion(
                url=self.api_url + "/v1/engines/" + self.engine + "/logprob",
                headers={"Authorization": "Bearer " + self.api_key},
                json={"context": context, "continuation": continuation},
            )
            resp = response.json()
            if "logprob" in resp:
                logprob = resp["logprob"]
                is_greedy = resp["is_greedy"]
                res.append((logprob, is_greedy))

                self.cache_hook.add_partial(
                    "loglikelihood", (context, continuation), (logprob, is_greedy)
                )
            else:
                logger.error(
                    f"The following response does not contain `logprobs`. Got:\n{resp}"
                )
                assert False
        return res

    def loglikelihood_rolling(self, requests):
        # TODO: The TextSynth API does not support tokenized inputs so we cannot
        # manually partition long contexts into smaller rolling windows as
        # done for other models derived from `BaseLM`. Override this method
        # with a windowing scheme that works for direct string inputs.
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported due to lack of "
            "input tokenization support from TextSynth."
        )

    def generate_until(self, requests):
        if not requests:
            return []

        requests_lst: List[Tuple[str, dict]] = [req.args for req in requests]
        
        res = []
        for request in tqdm(requests_lst):
            inp = request[0]
            request_args = request[1]
            # generation_kwargs
            until = request_args.get("until")
            temperature = request_args.get("temperature", self.temperature)
            temperature = temperature + 0.01 if temperature == 0 else temperature # temperature should not be absolute 0 
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            prediction_id = _requests.post(
                url=f"{self.api_url}/v1/predictions",
                headers=headers,
                json={
                    "version": self.version,
                    "input": {
                        "prompt": inp,
                        "max_new_tokens": self.max_gen_toks,
                        "temperature": temperature,
                        "stop": until, #TODO check if this works
                    },
                }
            ).json()['id']
            response = _requests.get(f"{self.api_url}/v1/predictions/{prediction_id}", headers=headers)
            while response.json()["status"] != "succeeded":
                time.sleep(2)
                response = _requests.get(f"{self.api_url}/v1/predictions/{prediction_id}", headers=headers)
            resp = response.json()
            if "output" in resp:
                s = ''.join(resp["output"])
                res.append(s)

                self.cache_hook.add_partial("generate_until", (inp, request_args), s)
            else:
                logger.error(
                    "The following response does not contain generated `text`. "
                    "Got:\n{resp}"
                )
                assert False
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

from utils import *


server = ServerPreset.tinyllama2()


def _single_token_bias(content: str, bias: float = 100.0) -> dict:
    global server
    res = server.make_request("POST", "/tokenize", data={
        "content": content,
    })
    assert res.status_code == 200
    assert len(res.body["tokens"]) >= 1
    return {str(res.body["tokens"][-1]): bias}


def test_reasoning_loop_guard_request_settings():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 1,
        "reasoning_loop_guard": "stop",
        "reasoning_loop_min_tokens": 768,
        "reasoning_loop_window": 1536,
        "reasoning_loop_max_period": 256,
        "reasoning_loop_min_coverage": 768,
        "reasoning_loop_check_interval": 16,
        "reasoning_loop_interventions": 0,
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 1
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"

    settings = res.body["generation_settings"]
    assert settings["reasoning_loop_guard"] == "stop"
    assert settings["reasoning_loop_min_tokens"] == 768
    assert settings["reasoning_loop_window"] == 1536
    assert settings["reasoning_loop_max_period"] == 256
    assert settings["reasoning_loop_min_coverage"] == 768
    assert settings["reasoning_loop_check_interval"] == 16
    assert settings["reasoning_loop_interventions"] == 0


def test_reasoning_loop_guard_off_keeps_hard_limit():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 2,
        "reasoning_loop_guard": "off",
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 2
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"
    assert res.body["generation_settings"]["reasoning_loop_guard"] == "off"


def test_reasoning_loop_guard_invalid_request_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 1,
        "reasoning_loop_guard": "force-close",
        "reasoning_loop_min_tokens": 768,
        "reasoning_loop_window": 768,
        "reasoning_loop_max_period": 512,
        "reasoning_loop_min_coverage": 768,
    })

    assert res.status_code == 400


def test_reasoning_loop_guard_stop_triggers_on_hidden_reasoning_loop():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Repeat a short hidden thought:",
        "n_predict": 16,
        "temperature": 0.0,
        "logit_bias": _single_token_bias(" loop"),
        "generation_prompt": "<think>",
        "reasoning_budget_start_tag": "<think>",
        "reasoning_budget_end_tag": "</think>",
        "reasoning_loop_guard": "stop",
        "reasoning_loop_min_tokens": 4,
        "reasoning_loop_window": 6,
        "reasoning_loop_max_period": 2,
        "reasoning_loop_min_coverage": 3,
        "reasoning_loop_check_interval": 1,
        "reasoning_loop_interventions": 0,
    })

    assert res.status_code == 200
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "reasoning_loop_guard"
    assert res.body["tokens_predicted"] < 16
    assert res.body["reasoning_tokens"] >= 4
    assert res.body["visible_completion_tokens"] == 0
    assert res.body["loop_guard"]["triggered"] is True
    assert res.body["loop_guard"]["action"] == "stop"


def test_reasoning_loop_guard_force_close_intervenes_then_continues():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Repeat a short hidden thought:",
        "n_predict": 12,
        "temperature": 0.0,
        "logit_bias": _single_token_bias(" loop"),
        "generation_prompt": "<think>",
        "reasoning_budget_start_tag": "<think>",
        "reasoning_budget_end_tag": "</think>",
        "reasoning_loop_guard": "force-close",
        "reasoning_loop_min_tokens": 4,
        "reasoning_loop_window": 6,
        "reasoning_loop_max_period": 2,
        "reasoning_loop_min_coverage": 3,
        "reasoning_loop_check_interval": 1,
        "reasoning_loop_interventions": 1,
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 12
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"
    assert res.body["reasoning_tokens"] >= 4
    assert res.body["visible_completion_tokens"] > 0
    assert res.body["loop_guard"]["triggered"] is True
    assert res.body["loop_guard"]["action"] == "force-close"

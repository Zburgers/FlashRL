import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flashrl.demo.server import DemoSession, create_server


@pytest.fixture
def live_demo():
    session = DemoSession(policy_name="rule", seed=23, max_episode_steps=20)
    server = create_server(session, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    session.start()
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", session
    finally:
        server.shutdown()
        server.server_close()
        session.close()
        thread.join(timeout=2)


def get_json(url):
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def post_json(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def test_demo_serves_status_frame_and_installed_assets(live_demo):
    base, _ = live_demo
    status = get_json(f"{base}/api/status")
    frame = get_json(f"{base}/api/frame")
    assert status["environment_id"] == "FlashRL-DinoSim-v2"
    assert status["policy"]["name"] == "rule"
    assert set(frame) >= {
        "episode",
        "dino",
        "obstacles",
        "score",
        "action",
        "reward",
        "q_values",
    }
    assert frame["q_values"] is None
    with urlopen(f"{base}/", timeout=2) as response:
        html = response.read().decode()
    assert "FlashRL" in html
    assert "canvas" in html


def test_demo_reset_pause_and_speed_controls(live_demo):
    base, _ = live_demo
    reset = post_json(f"{base}/api/reset", {"seed": 99})
    assert reset["seed"] == 99
    paused = post_json(f"{base}/api/control", {"paused": True, "speed": 2})
    assert paused["paused"] is True
    assert paused["speed"] == 2


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/reset", {"seed": "not-an-integer"}),
        ("/api/reset", {"seed": 2**40}),
        ("/api/control", {"speed": 100}),
        ("/api/control", {"checkpoint": "../../arbitrary.pt"}),
    ],
)
def test_demo_rejects_invalid_or_remote_checkpoint_controls(live_demo, path, payload):
    base, _ = live_demo
    with pytest.raises(HTTPError) as failure:
        post_json(f"{base}{path}", payload)
    assert failure.value.code == 400

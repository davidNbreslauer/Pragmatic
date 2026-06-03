import json

from pragmatic.cli import main
from pragmatic.schemas import ResearchState, Thesis


ARBITRARY_THESIS = "Room-temperature superconductors are ready for commercial grid storage"


def test_cli_run_blocks_arbitrary_prepared_prompt(capsys):
    exit_code = main(["run", "--thesis", ARBITRARY_THESIS])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "spider-silk demo corpus" in captured.err
    assert "--source-mode web" in captured.err


def test_cli_run_supports_arbitrary_live_web_mode(monkeypatch, tmp_path):
    captured = {}

    def fake_run_research_loop(thesis_text, **kwargs):
        captured["thesis_text"] = thesis_text
        captured.update(kwargs)
        return ResearchState(thesis=Thesis(text=thesis_text))

    monkeypatch.setattr("pragmatic.cli.run_research_loop", fake_run_research_loop)
    output_path = tmp_path / "state.json"

    exit_code = main(
        [
            "run",
            "--thesis",
            ARBITRARY_THESIS,
            "--source-mode",
            "web",
            "--allow-live-web-search",
            "--web-search-model",
            "test-search-model",
            "--max-web-sources",
            "3",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["thesis"]["text"] == ARBITRARY_THESIS
    assert captured["source_mode"] == "web"
    assert captured["allow_live_web_search"] is True
    assert captured["web_search_model"] == "test-search-model"
    assert captured["max_web_sources"] == 3

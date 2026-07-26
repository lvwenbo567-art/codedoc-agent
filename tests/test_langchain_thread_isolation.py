from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from test_langchain_short_term_memory import FakeMemoryAgent, build_service


def test_different_thread_isolated(tmp_path):
    agent = FakeMemoryAgent()
    service = build_service(tmp_path, agent)

    service.run(
        "The project uses a rerank client named RerankClient.",
        project_id=1,
        thread_id="thread-A",
    )
    result = service.run(
        "Which file is it in?",
        project_id=1,
        thread_id="thread-B",
    )

    assert result.effective_thread_id == "project:1:thread:thread-B"
    assert "Insufficient evidence" in result.answer


def test_same_thread_id_different_project_isolated(tmp_path):
    agent = FakeMemoryAgent()
    service = build_service(tmp_path, agent)

    service.run(
        "The project uses a rerank client named RerankClient.",
        project_id=1,
        thread_id="shared-thread",
    )
    result = service.run(
        "Which file is it in?",
        project_id=2,
        thread_id="shared-thread",
    )

    assert result.effective_thread_id == "project:2:thread:shared-thread"
    assert "Insufficient evidence" in result.answer

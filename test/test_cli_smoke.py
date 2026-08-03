from __future__ import annotations


def test_python_module_main_functions(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    from src.data_ingestion import main as ingestion_main
    from src.ml_models import main as ml_main
    from src.statistical_models import main as statistics_main
    from src.visualization import main as visualization_main

    ingestion_main()
    statistics_main()
    ml_main()
    visualization_main()
    output = capsys.readouterr().out
    assert "successful" in output
    assert "Created" in output

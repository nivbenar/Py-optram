from importlib import import_module

wrapper = import_module("pyoptram.optram")


def test_optram_orchestrates_r_workflow(monkeypatch, tmp_path, capsys):
    s2_dir = tmp_path / "s2"
    data_dir = tmp_path / "data"
    vi_dir, str_dir = s2_dir / "NDVI", s2_dir / "STR"
    vi_dir.mkdir(parents=True)
    str_dir.mkdir()
    for path in (vi_dir / "z.tif", vi_dir / "a.tif", str_dir / "y.tif", str_dir / "b.tif"):
        path.touch()
    calls = []
    table = object()
    rmse = object()
    monkeypatch.setattr(wrapper, "acquire_optram_inputs", lambda **kwargs: calls.append(("acquire", kwargs)))
    monkeypatch.setattr(
        wrapper,
        "optram_ndvi_str",
        lambda vi, st, **kwargs: calls.append(("table", vi, st, kwargs)) or table,
    )
    monkeypatch.setattr(
        wrapper,
        "optram_wetdry_coefficients",
        lambda value, **kwargs: calls.append(("fit", value, kwargs)) or rmse,
    )

    result = wrapper.optram("aoi", "2024-01-01", "2024-01-03", s2_dir, data_dir)

    assert calls[0] == ("acquire", {"aoi": "aoi", "from_date": "2024-01-01", "to_date": "2024-01-03", "output_dir": s2_dir})
    assert calls[1] == ("table", [vi_dir / "a.tif", vi_dir / "z.tif"], [str_dir / "b.tif", str_dir / "y.tif"], {"output_csv": data_dir / "VI_STR_data.csv", "features": "aoi"})
    assert calls[2] == ("fit", table, {"output_dir": data_dir})
    assert result is rmse
    assert "RMSE for fitted trapezoid:" in capsys.readouterr().out


def test_optram_uses_default_temp_directories(monkeypatch, tmp_path):
    (tmp_path / "NDVI").mkdir()
    (tmp_path / "STR").mkdir()
    calls = []
    rmse = object()
    monkeypatch.setattr(wrapper.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(wrapper, "acquire_optram_inputs", lambda **kwargs: calls.append(("acquire", kwargs)))
    monkeypatch.setattr(wrapper, "optram_ndvi_str", lambda *args, **kwargs: calls.append(("table", args, kwargs)) or "table")
    monkeypatch.setattr(wrapper, "optram_wetdry_coefficients", lambda *args, **kwargs: calls.append(("fit", args, kwargs)) or rmse)

    assert wrapper.optram("aoi", "2024-01-01", "2024-01-03") is rmse
    assert calls[0][1]["output_dir"] == str(tmp_path)
    assert calls[1] == ("table", ([], []), {"output_csv": tmp_path / "VI_STR_data.csv", "features": "aoi"})
    assert calls[2] == ("fit", ("table",), {"output_dir": str(tmp_path)})

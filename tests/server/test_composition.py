from __future__ import annotations


def test_factory_passes_concrete_components_to_the_engine(monkeypatch) -> None:
    from nari_qwen3_tts import factory

    model = object()
    assets = object()
    executor = object()
    pipeline = object()
    catalog = object()
    engine_config = object()
    captured = {}
    product = object()

    monkeypatch.setattr(
        factory,
        "_build_engine_components",
        lambda *args, **kwargs: (executor, pipeline, catalog),
    )
    monkeypatch.setattr(factory, "CaptureCatalog", type(catalog))

    def construct(received_model, **kwargs):
        captured.update(model=received_model, **kwargs)
        return product

    monkeypatch.setattr(factory, "Engine", construct)

    assert factory.build_qwen3_tts_engine(
        model,
        assets,
        engine_config=engine_config,
        capture=False,
    ) is product
    assert captured == {
        "model": model,
        "executor": executor,
        "pipeline": pipeline,
        "capture_catalog": catalog,
        "config": engine_config,
    }

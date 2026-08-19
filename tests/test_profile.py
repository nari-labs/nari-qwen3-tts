from __future__ import annotations


def test_resolved_profile_exposes_policy_stage_and_resource_values() -> None:
    from nari_qwen3_tts.profile import ProfileLoader

    resolved = ProfileLoader().load_profile("ttfa")

    assert resolved.name == "ttfa"
    assert len(resolved.sha256) == 64
    assert resolved.stages.talker_decode.max_batch_size == 32
    assert resolved.resources.kv_pages == 256


def test_profile_loader_resolves_a_yaml_overlay_against_its_base(tmp_path) -> None:
    from nari_qwen3_tts.profile import ProfileLoader

    loader = ProfileLoader()
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "extends: ttfa\ntalker_decode:\n  max_batch_size: 24\n"
        "  batch_sizes: [1, 2, 4, 8, 12, 16, 24]\n",
        encoding="utf-8",
    )
    loaded = loader.load_yaml(overlay)

    assert loaded.stages.talker_decode.max_batch_size == 24
    assert loaded.resources.kv_pages == loader.load_profile("ttfa").resources.kv_pages


def test_lifecycle_live_input_and_wire_settings_carry_their_own_values() -> None:
    from nari_qwen3_tts.config import ApiConfig, EngineConfig, LiveInputConfig

    live = LiveInputConfig(max_update_tokens=17)
    engine = EngineConfig(live_input=live)
    api = ApiConfig(command_timeout_s=3.0)

    assert engine.live_input is live
    assert engine.max_buffered_pcm_bytes == 8 * 1024 * 1024
    assert api.command_timeout_s == 3.0

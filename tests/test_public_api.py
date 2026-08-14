import mc_ambient


def test_public_api_exports_core_workflow_functions():
    for name in [
        "load_audio",
        "save_audio",
        "stft_mc",
        "istft_mc",
        "estimate_scm",
        "estimate_time_varying_scm",
        "synthesize_static",
        "synthesize_time_varying",
        "apply_multichannel_rir",
        "complex_coherence",
    ]:
        assert callable(getattr(mc_ambient, name))

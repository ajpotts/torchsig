
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.signal_building import signal_generator_lookup_table


def test_signal_generator_keys_match_class_family():
    # CLASS_FAMILY_DICT.keys() should be a subset of signal_generator_lookup_table.keys()
    assert set(signal_generator_lookup_table.keys()) >= set(CLASS_FAMILY_DICT.keys())


def test_torchsig_signal_lists_groups_expected_signal_families():
    signal_lists = TorchSigSignalLists()

    assert "fm" in signal_lists.fm_signals
    assert "tone" in signal_lists.tone_signals
    assert "chirpss" in signal_lists.chirpss_signals

    assert all("ofdm" in name for name in signal_lists.ofdm_signals)
    assert all(
        any(key in name for key in ["fsk", "msk"])
        for name in signal_lists.fsk_signals
    )
    assert all(
        any(key in name for key in ["ask", "qam", "psk", "ook"])
        for name in signal_lists.constellation_signals
    )
    assert all("am-" in name for name in signal_lists.am_signals)
    assert all("lfm-" in name for name in signal_lists.lfm_signals)


def test_torchsig_signal_lists_assigns_every_known_signal_to_at_most_one_group():
    signal_lists = TorchSigSignalLists()

    grouped_signals = (
        signal_lists.fsk_signals
        + signal_lists.ofdm_signals
        + signal_lists.constellation_signals
        + signal_lists.am_signals
        + signal_lists.fm_signals
        + signal_lists.lfm_signals
        + signal_lists.chirpss_signals
        + signal_lists.tone_signals
    )

    assert len(grouped_signals) == len(set(grouped_signals))


def test_torchsig_signal_lists_grouped_signals_are_known_signals():
    signal_lists = TorchSigSignalLists()

    grouped_signals = (
        signal_lists.fsk_signals
        + signal_lists.ofdm_signals
        + signal_lists.constellation_signals
        + signal_lists.am_signals
        + signal_lists.fm_signals
        + signal_lists.lfm_signals
        + signal_lists.chirpss_signals
        + signal_lists.tone_signals
    )

    assert set(grouped_signals).issubset(set(signal_lists.all_signals))


def test_torchsig_signal_lists_repeated_init_does_not_duplicate_entries():
    first = TorchSigSignalLists()
    first_grouped_count = sum(
        len(group)
        for group in [
            first.fsk_signals,
            first.ofdm_signals,
            first.constellation_signals,
            first.am_signals,
            first.fm_signals,
            first.lfm_signals,
            first.chirpss_signals,
            first.tone_signals,
        ]
    )

    second = TorchSigSignalLists()
    second_grouped_count = sum(
        len(group)
        for group in [
            second.fsk_signals,
            second.ofdm_signals,
            second.constellation_signals,
            second.am_signals,
            second.fm_signals,
            second.lfm_signals,
            second.chirpss_signals,
            second.tone_signals,
        ]
    )

    assert second_grouped_count == first_grouped_count

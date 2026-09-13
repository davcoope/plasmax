"""Initialization capture, calibration, and canonical YAML export behavior."""

from pathlib import Path

import numpy as np
import pytest

from plasmax.environment.initialization_data import (
    ToraxInitialization,
    load_initialization,
    round_significant,
    write_initialization,
)
from plasmax.environment.registry import CONFIGS_DIR
from tools.calibration.improve_initial_conditions import (
    fit_gaussian_moments,
    reconstruct_psi_from_q,
    write_itpa_initialization,
)
from tools.generate_phase_initialization import Config, capture, main


@pytest.mark.parametrize("q_value", (0.7, 3.2))
def test_constant_q_reconstructs_quadratic_poloidal_flux(q_value: float) -> None:
    rho = np.asarray([0.0, 0.03, 0.17, 0.43, 0.76, 1.0])
    edge_psi = 7.0

    psi = reconstruct_psi_from_q(rho, np.full_like(rho, q_value), edge_psi)

    np.testing.assert_allclose(psi, edge_psi * rho**2, rtol=1e-14, atol=0.0)


def test_varying_q_reconstructs_logarithmic_poloidal_flux() -> None:
    rho = np.linspace(0.0, 1.0, 1001)
    edge_psi = 7.0

    psi = reconstruct_psi_from_q(rho, 1.0 + rho**2, edge_psi)

    # Integrating rho / (1 + rho**2) gives log(1 + rho**2) / 2.
    # The tolerance allows the numerical quadrature error on this grid.
    expected = edge_psi * np.log1p(rho**2) / np.log(2.0)
    np.testing.assert_allclose(psi, expected, rtol=1e-6, atol=1e-9)


def test_q_reconstruction_normalizes_to_and_scales_with_edge_flux() -> None:
    rho = np.asarray([0.02, 0.14, 0.4, 0.65, 0.98])
    q = 1.0 + 3.0 * rho**2
    edge_psi = 2.0
    scale = 3.5

    psi = reconstruct_psi_from_q(rho, q, edge_psi)
    scaled = reconstruct_psi_from_q(rho, q, scale * edge_psi)

    np.testing.assert_allclose(psi[-1], edge_psi, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(scaled, scale * psi, rtol=1e-14, atol=0.0)


def test_gaussian_fit_matches_volume_weighted_centroid_and_rms_width() -> None:
    rho = np.linspace(0.02, 0.98, 25)
    # The shaped volume measure makes Gaussian input parameters differ from
    # the physical dV-weighted moments that the fitter must reproduce.
    measure = 1.0 + 3.0 * rho
    location, width = fit_gaussian_moments(
        rho, measure, target_centroid=0.55, target_rms_width=0.18
    )
    profile = np.exp(-0.5 * np.square((rho - location) / width))
    weights = profile * measure
    centroid = np.sum(rho * weights) / np.sum(weights)
    rms_width = np.sqrt(np.sum(np.square(rho - centroid) * weights) / np.sum(weights))

    np.testing.assert_allclose(
        (centroid, rms_width), (0.55, 0.18), rtol=0.0, atol=1e-12
    )


def test_zero_step_capture_exports_the_selected_nominal_state(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    main(Config(environment="iter/hybrid/flattop", output=path))
    selected = CONFIGS_DIR / "data/initializations/iter/hybrid/settled.yaml"
    assert path.read_bytes() == selected.read_bytes()


def test_kstar_capture_reproduces_the_packaged_rounded_history(tmp_path: Path) -> None:
    document = capture(Config(environment="kstar_worldmodel"))
    path = tmp_path / "kstar.yaml"
    write_initialization(document, path)
    actual = load_initialization(path, kind="kstar")
    expected = load_initialization(
        CONFIGS_DIR / "data/initializations/kstar/nominal.yaml", kind="kstar"
    )
    assert actual.inputs == expected.inputs
    assert actual.history_row == expected.history_row
    assert actual.targets == expected.targets


def test_calibration_exports_complete_rounded_yaml(tmp_path: Path) -> None:
    source = np.genfromtxt(
        CONFIGS_DIR / "data/references/iter_baseline_450s_profiles_25.csv",
        delimiter=",",
        names=True,
    )
    columns = {
        "rho": "rho",
        "T_i": "T_i_keV",
        "T_e": "T_e_keV",
        "n_e": "n_e_m3",
        "psi": "psi_Wb",
        "q": "q",
    }
    profiles = {name: source[column] for name, column in columns.items()}
    path = tmp_path / "itpa.yaml"
    write_itpa_initialization(profiles, path)
    document = load_initialization(path, kind="torax")
    assert isinstance(document, ToraxInitialization)
    for field, column in (
        ("T_i_keV", "T_i"),
        ("T_e_keV", "T_e"),
        ("n_e_m3", "n_e"),
        ("psi_Wb", "psi"),
    ):
        np.testing.assert_array_equal(
            getattr(document.profiles, field), round_significant(profiles[column])
        )
    assert document.provenance.reference == "iter_baseline_hot_450s"
    assert set(tmp_path.iterdir()) == {path}

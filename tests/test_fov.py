import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.data.fov import fit_circle, fov_geometry
from synthetic import make_fundus


def test_full_circle_is_recovered():
    fov = fov_geometry(make_fundus(width=800, height=600, r=250, cx=410, cy=290))
    assert fov.shape_class == "full"
    assert (fov.cx, fov.cy, fov.r) == pytest.approx((410, 290, 250), abs=1.0)
    assert fov.truncation == 0


def test_vertically_cut_circle_is_recovered_beyond_the_frame():
    fov = fov_geometry(make_fundus(width=800, height=600, r=340))
    assert fov.shape_class == "cut_vertical"
    assert fov.r == pytest.approx(340, rel=0.01)
    expected = 1 - 2 * math.asin(300 / 340) / math.pi
    assert fov.truncation == pytest.approx(expected, abs=0.01)


def test_circle_cut_on_all_sides():
    fov = fov_geometry(make_fundus(width=600, height=600, r=330))
    assert fov.shape_class == "cut_both"


def test_black_image_fails():
    assert fov_geometry(np.zeros((300, 400, 3), dtype=np.uint8)).shape_class == "failed"


def test_rectangle_is_irregular():
    rgb = np.zeros((300, 400, 3), dtype=np.uint8)
    rgb[50:250, 60:340] = 200
    assert fov_geometry(rgb).shape_class == "irregular"


@given(
    cx=st.floats(-200, 200),
    cy=st.floats(-200, 200),
    r=st.floats(20, 500),
    start=st.floats(0, 2 * math.pi),
    span=st.floats(math.pi / 2, 2 * math.pi),
)
def test_fit_circle_recovers_arcs(cx, cy, r, start, span):
    theta = np.linspace(start, start + span, 200)
    noise = np.random.default_rng(0).normal(scale=0.2, size=theta.size)
    xs, ys = cx + (r + noise) * np.cos(theta), cy + (r + noise) * np.sin(theta)
    fx, fy, fr, rms = fit_circle(xs, ys)
    assert (fx, fy, fr) == pytest.approx((cx, cy, r), abs=0.02 * r + 0.5)
    assert rms < 1.0

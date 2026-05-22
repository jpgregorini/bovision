import numpy as np
import pytest
from src.measure import MorphologyMeasurer, Measurements


@pytest.fixture
def rectangular_mask():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[140:340, 170:470] = 1
    return mask


@pytest.fixture
def uniform_depth():
    return np.full((480, 640), 3.0, dtype=np.float32)


def test_measure_returns_measurements(rectangular_mask, uniform_depth):
    measurer = MorphologyMeasurer()
    result = measurer.measure(rectangular_mask, uniform_depth)
    assert isinstance(result, Measurements)
    assert result.comprimento_m > 0
    assert result.altura_m > 0
    assert result.largura_m > 0
    assert result.area_m2 > 0
    assert result.perimetro_m > 0


def test_measure_larger_mask_gives_larger_measurements(uniform_depth):
    measurer = MorphologyMeasurer()
    small_mask = np.zeros((480, 640), dtype=np.uint8)
    small_mask[200:280, 250:370] = 1
    large_mask = np.zeros((480, 640), dtype=np.uint8)
    large_mask[100:380, 120:520] = 1
    small_m = measurer.measure(small_mask, uniform_depth)
    large_m = measurer.measure(large_mask, uniform_depth)
    assert large_m.comprimento_m > small_m.comprimento_m
    assert large_m.altura_m > small_m.altura_m
    assert large_m.area_m2 > small_m.area_m2


def test_measure_closer_depth_gives_smaller_real_measurements():
    measurer = MorphologyMeasurer()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[140:340, 170:470] = 1
    depth_close = np.full((480, 640), 2.0, dtype=np.float32)
    depth_far = np.full((480, 640), 5.0, dtype=np.float32)
    m_close = measurer.measure(mask, depth_close)
    m_far = measurer.measure(mask, depth_far)
    assert m_far.comprimento_m > m_close.comprimento_m


def test_measurements_dataclass():
    m = Measurements(comprimento_m=1.43, altura_m=1.31, largura_m=0.49, area_m2=1.92, perimetro_m=1.84)
    assert m.comprimento_m == 1.43
    assert m.area_m2 == 1.92


def test_empty_mask_raises():
    measurer = MorphologyMeasurer()
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        measurer.measure(empty_mask, depth)

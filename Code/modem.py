import numpy as np

def RRC(sps, roll_off, span):
    half_span = span // 2
    time_axis = np.arange(-half_span * sps, half_span * sps + 1) / sps

    numerator = np.sin(np.pi * time_axis * (1 - roll_off)) + (
        4 * roll_off * time_axis * np.cos(np.pi * time_axis * (1 + roll_off)))
    denominator = np.pi * time_axis * (1 - (4 * roll_off * time_axis) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rrc = numerator / denominator

    rrc[np.isclose(time_axis, 0)] = 1 - roll_off + (4 * roll_off) / np.pi
    rrc[np.isclose(np.abs(time_axis), (1/(4 * roll_off)))] = (roll_off / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * roll_off)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off)))

    return rrc / np.sqrt(np.sum(rrc ** 2))


def demapper(mapped_symbols):
    real_nums = mapped_symbols.real
    imaginary_nums = mapped_symbols.imag

    first_bit = (imaginary_nums < 0).astype(int)
    second_bit = (real_nums < 0).astype(int)
    demapped = np.column_stack((first_bit, second_bit))
    demapped_bits = demapped.flatten()
    return demapped_bits
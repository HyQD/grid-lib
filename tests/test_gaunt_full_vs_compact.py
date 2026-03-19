import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
    setup_y_and_ybar_compact,
)


def test_gaunt_full_vs_compact():
    l_max = 3
    m_max = l_max

    L_max = 2 * l_max
    M_max = 2 * m_max

    n_lm = number_of_lm_states(l_max, m_max)
    n_LM = number_of_lm_states(L_max, M_max)

    y, y_bar = setup_y_and_ybar_sympy(l_max, m_max, L_max, M_max)
    y_c, y_bar_c = setup_y_and_ybar_compact(l_max, m_max, L_max, M_max)

    for L in range(L_max + 1):
        for m1 in range(-m_max, m_max + 1):
            for l1 in range(abs(m1), l_max + 1):
                I_l1m1 = LM_to_I(l1, m1, l_max, m_max)
                for m2 in range(-m_max, m_max + 1):
                    for l2 in range(abs(m2), l_max + 1):
                        I_l2m2 = LM_to_I(l2, m2, l_max, m_max)

                        M12 = m1 - m2
                        M21 = m2 - m1

                        I_LM12 = LM_to_I(L, M12, L_max, M_max)
                        I_LM21 = LM_to_I(L, M21, L_max, M_max)

                        if abs(M12) <= L_max:
                            assert np.allclose(
                                y[I_LM12, I_l1m1, I_l2m2], y_c[L, I_l1m1, I_l2m2]
                            )
                        if abs(M21) <= L_max:
                            assert np.allclose(
                                y_bar[I_LM21, I_l1m1, I_l2m2],
                                y_bar_c[L, I_l1m1, I_l2m2],
                            )

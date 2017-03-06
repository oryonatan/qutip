import warnings

import pyximport

warnings.filterwarnings('ignore')

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
from qutip import *
import numpy as np

pyximport.install(setup_args={"include_dirs": np.get_include()})
import XXZZham as XXZZham
from XXZZham import add_high_energies, rotate_to_00_base
import random
import adiabatic_sim as asim
import time

import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(4)))
import os

n = 4

while True:
    def create_vector_from_string(vec_to_build: str) -> Qobj:
        """ Creates a vector from 0/1 string - indicator in the computational basis
            i.e. the string 010 will create the vector |010>

        """
        to_tensor = []
        for digit in vec_to_build:
            if digit == '0':
                to_tensor.append(basis(2, 0))
            elif digit == '1':
                to_tensor.append(basis(2, 1))
            else:
                raise ValueError("String should consist only of 0 and 1")
        return tensor(to_tensor)


    data:image / png;
    base64, iVBORw0KGgoAAAANSUhEUgAAA20AAAFCCAYAAABilzUAAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1 + / AAAIABJREFUeJzs3XmYHFW5 + PHvW1XdPftkJstkD2sk7DthEUJEdgXZVDYRWRS8 / Lig94KggOAGIrJeUAGJLHoBERWReJEAguw7BMISspA9M5m9u2s5vz9O9UzPZLaezBbyfp7nPOfUOaeqTrcyJ2 / XqSoxxqCUUkoppZRSamRyhnsASimllFJKKaW6p0GbUkoppZRSSo1gGrQppZRSSiml1AimQZtSSimllFJKjWAatCmllFJKKaXUCKZBm1JKKaWUUkqNYBq0KRUTkUYR2Wy4x9EbEYlEZIt + 7
    rtQRGZ307afiMzvqq + IXCwiv + rhuCeKyN / 7
    MyallFIjw8bwt1xEnhCR04d7HEoNNQ3a1IAQkdNE5A0RaRaRZSJyi4hUDve4utPVH31jTLkx5uNhGlIhBuXlisaYfxljZnTT9hNjzFkAIjItDhydvPZ7jTGHDsa4lFJKDSwR + VhEWkSkIf7BskFEbtC / 5
    UqNXBq0qQ0mIhcCPwEuBCqAmcA04B8i4g3n2DY2IuL2pdugD6T385sRMA6llFL9Y4AjjDEV8Q + WFcaY84Z7UEqp7mnQpjaIiJQDlwPfNsb8wxgTGmMWAycAmwEnx / 0
    cEfmeiHwgIvUi8qKITIrbthORuSKyVkSWi8hFcf2dIvLDvHMdICJL8rYXishFIvJ2vO / tIpKM20aJyF9EZFXc9hcRmRi3XQV8Frgp9 + tiXN + 27
    FBEKkRkTrz / QhG5JO + 8
    XxORp0XkGhGpFZEPRaTbXyZ7GecBIrJERP5LRJYDd8T1Z4rI + yKyRkT + JCITOh32iPi8q0Tk6rxzbSEij8f7rRKRu0WkotO + e / Y0lm4 + w2UiMifefDLO18Xf31657ySv / zZ5 / 5
    vOF5Hj89oOj8 / fEH / 2
    C7r77pRSSg2a9X546 + Jv + cEi8q6I1InIzSIyL3 + VioicLiLvxH / rHxWRqXltkYicLSIL4rnyprg + GR9v27y + Y + Irf2O6mb8ndfkB7Nz0u7ztDitB4rn8N2JXAC0RkStFROK2LePPsy4 + 130
    b9nUqNbg0aFMbah8gBTyUX2mMaQb + Bnw + rroQ + DJwqDGmEjgdaBGRMuAfcd8JwFbA4z2cr / PSwBPjc2wJfAa4NK53sAHQFGAq0ALcHI / tUuBpbKCZ / +ti / rFvAsqxgecs4FQR + Xpe + 57
    AfGA0cA1wew9j7mmcAOOBUfE4zxJ7H9mPgeOw38li4Pedjnc0sGucjsqbRCXedzwwA5iMDar7Opa + LL3cP84r4u / v + fx9RaQEmAvcDYwBvgLcIiLbxP1 + A5xpjKkAtgf + 2
    YdzKqWUGhq5v + VjgPuB / 8
    bOde8Be + c6ichRwEXY + Wgsdl7tHPgcAewG7AScICIHG2OywIPAV / P6nQDMM8asoev5 + 6
    bextvN9l1AFtgC2AU7950Rt10JPGaMGYWdK2 / s4RxKDTsN2tSGGgOsMcZEXbQtj9sBvgFcYoz5AMAY86Yxpg44ElhujPmlMSZrjGk2xrxYwPlvNMYsM8asA35EPAkYY2qNMQ8ZYzJxAPkT2oON7uR + fXOwAeZFxpgWY8wi4FrglLy + i4wxdxhjDHZSGC8i4wodZywELjPG + MaYDDaout0Y87oxxgcuBvbO / wUT + Kkxpt4YsxT4Zd7n / tAY87gxJjDGrAWuAw4oYCyF6G555JHAQmPMHGO9jp2gc1fbssB2IlIef4bX + nl + pZRS / fen + ApYXZx / o1P7YcBbxpiHjTGRMeYGYGVe + 9
    nAT4wxC + J / A / wU2FlEpuT1 + YkxptEYswR4Atg5rr + PjnPPicC90O / 5
    ez0iUhN / hv80xqTjgPCX2B8SAXxgmohMiv / 98
    Wyh51BqKGnQpjbUGmCM5D2UIs + EuB3sL2YfddFnCvDhBpx / aV55EZBbAlksIreJvdl6HXZJ36jcsohejAE87BWu / GPnL89YkSsYY1qxAUxZoeOMrY6Ds5yJcZ / c8ZuBtZ3O393nHici94nI0vhz56529XUsA2EaMDP + R0CtiNRhJ + SauP1Y7K + vi8Q + EGbmAJ9fKaVU744yxlQbY6rivPOKkYlA5yXz + fPHNOD63N967Dxl6DhX5Qd5LbTPk08AxSKyh4hMw16Jewg2eP7ONxVIAMvz5qJbsVcFAb6L / XfwCyLyZqfVNEqNOBq0qQ31byADHJNfGS97PAz4v7hqCXY5Xmfd1QM0AyV5253v6wIb9OVMA5bF5e8AWwN7xEsfcr / S5f7o97QMcA3xL3Cdjv1JD / v0prtxdjWWZfnnFpFS7NKU / Mmyu + P9BIiA7eLPfTLrXxHraSx90dsSyiXYZS7Vef8gqDDGfBvAGPOyMSa3nOZh4H8LPL9SSqkN11sQtJyO8wXYZYQ5S4CzO / 2
    tLzPGPNfbieMrc / +L / UHvq8Bf4x8owd5O0dP8na + nfycsAdLA6LzxjTLG7BiPYZUx5ixjzCTgm9hl / P16nY5SQ0GDNrVBjDENwA + BG0XkEBHxxL7r7A / YK1V3x11 / A1wpIlsBiMgOIlIF / BW7tPC8 + ObkMhHZM97nNeBwEakSkfHA / +tiCOeKyCQRqQa + R / u9X2VAK9AQt13eab + V2DXuXX2m3GTyo3g804D / BH7XVf8 + 6
    m6cXbkP + LqI7CgiKew9as / Fy0tyvhvfrD0FOI + On7sJaIxv3P7uBo6lK6uxgWF3wfZfgekicnL8 / 4
    eEiOwu9uEkCbHvAaowxoRAI3Z5qFJKqZHlEWB7EfmiiLgi8m3aV0yAvWr1vdwDRUSkUkSOK + D492FvRWhbGhkrp + f5O99rwP4iMkXsa4YuyjUYY1Zg76 + +TkTKxdpCRPaPx3tc3gNO1mHnta5u9VBqRNCgTW0wY8w12H / 8 / xyox159WwQclLfs7xfYQGiuiNRjg7hiY0wT9sbgL2KXHC7APvgDbJD0BvAx8He6Di7uxf5R / gB4H3uPFth16yXYq2bPYh90ku964Pj4yVS / zH2UvPbzsEs5PgKeAu42xtzZ09fQQ1tP41z / QMY8Dnwf + CP26t7mtK / Bz53rYeBl4BXgL8RPnQSuwN70vS6uf7CLcfZ1LF1 + png56I + AZ + IlJ3t2am8CDo7HvCxOPwWScZdTgIXxspezsBO2UkqpofUX6fietgfJ + 7
    sf3xd9PPZhW2uAbYCXsKtrMMb8Cfu3 / ffx3 / M3gPwnKff0gBCMMS9gr5RNAB7Na + pt / s4f4 / 9
    hfyR + A3gRO + / lOxU797wD1GIfrDI + btsDeF5EGoA / AeeZjeNdrWoTJfY5CkptfERkIfANY8yIfvrgxjJOpdTgEJHJwBzsVYoI + HX8UAelNhrxPWVLgRONMU / 21l
    8
    pNbD0SptSSik1uALgAmPMdthHpp + b9woMpUYsse9pq4yX6ufeV9rrPWtKqYGnQZvamG0sl4k3lnEqpQaBMWZF7tUW8fLh + XR8wp5SI9Xe2Cc8r8I + 9
    feo + NU0SqkhpssjlVJKqSESP6hpHrB9HMAppZRSvdIrbUoppdQQiF + F8gDw / zRgU0opVQhvKE8mInpZTymlNiHGmEJfiPupJCIeNmD7nTHm4W766ByplFKbiELnxyG / 0
    maM0dTHdNlllw37GDampN + Xfl / 6
    nY2spDq4A3jHGHN9T52G + 3 + zjSnpf4 / 6
    fen3NXKSfl + Fpf7Q5ZFKKaXUIBKRfYGTgNki8qqIvCIih / a2n1JKKZUzpMsjlVJKqU2NMeYZwB3ucSillNp46ZW2EWzWrFnDPYSNin5fhdHvq3D6nSk1cuh / j4XR76sw + n0VRr + vwTekj / wXETOU51NKKTV8RASjDyLpM50jlVJq09Cf + VGvtCmllFJKKaXUCKZBm1JKKaWUUkqNYBq0KaWUUkoppdQIpkGbUkoppZRSSo1gGrQppZRSSiml1AimQZtSSimllFJKjWAatCmllFJKKaXUCKZBm1JKKaWUUkqNYBq0KaWUUkoppdQIpkGbUkoppZRSSo1gGrQppZRSSiml1AimQZtSSimllFJKjWAatCmllFJKKaXUCKZBm1JKKaWUUkqNYBq0KaWUUkoppdQI1mvQJiK3i8hKEXmjhz43iMj7IvKaiOw8sENUSimllFJKqU1XX6603Qkc0l2jiBwGbGmM2Ro4G7h1gMamlFJKKaWUUpu8XoM2Y8y / gLoeuhwFzIn7Pg9UikjNwAxPKaWUUkoppTZt3gAcYxKwJG / 7
    k7huZVed // d / B + CMg0RkuEcw8PrzmYZqn / 7
    o6TzdtY2EfQZb53N3NRaR9vquyr219TV1x5iex + 84
    Nrlue7mnlH + u / PF29R3kyp4HiUR7cvSuXqWUUkptBAYiaCvIFVdc3lYeO3YW48bNGuohdKmnf1BurPrzmYZqn / 7
    o6TzdtY2EfXo61kAEep3P3dVYjGmv76rcW1shqT8BrzEQRRCGNu8theH6nzf / c3dVNsbul82C79vkuh2DuGRy / e2yMhg92qbq6vZy5 + 3
    qaigp + XT + +NNX8 + bNY968ecM9jI1baen6v4TkfqXorb6rfj316a3cU11f2vLz / tT1tdzVdk + / 9
    nRV39dfi3rq31tdruy6Hcu95Z37u + 6
    m / YdGqU2UmD78a1NEpgF / Mcbs2EXbrcATxpg / xNvvAgcYY9a70iYipi / nU0qpwWYMBEF7AOf7HQO63HZjI9TWwtq17Sl / O79sjA3gxo2DSZO6T1VVm8a / uUQEY8wm8EkHhogY09jY9a8hUTQwdZ3bu2obiLrcLyz57T31yeVh2P0 + 3
    ZW72qenlH + eQn8tyu3bVVtXdfn75PoUknfeL1fOBZ + dgzrP61jXOXXVnqsrJO9vSiTWz7uq67wsoqtf1BKJTeMPqfpU6s / 82
    NcrbRKnrvwZOBf4g4jMBNZ1FbAppdRIItI + / w + UlhYbxK1cCZ980p7 + 9
    a + O29ksTJzYMZAbOxZSKftvkfy8q7pcPnasvcKnPkXKyoZ7BGqkywWo3QV1QdCxrXPqqj1X15e8c7lzSqfXr / P9juXcdld553J3v6gFQftyiVwQl0x2n3J / PLurLypq / 6
    Pb13JxcXvKbXtDvohNbSJ6vdImIvcCs4DR2PvULgOSgDHG / CrucxNwKNAMfN0Y80o3x9IrbUqpTV5zc8cg7pNP7JW6TMb + WyST6Vjurm7lSjj + eLjoIthyy + H + VOvTK22F0TlSqQLkL5fIBXX5eS7l / mB2Tp3 / oKbT7X9ouyvnb6fT0NrannLbjtMxiOucSkvbU0lJx + 2
    u6svKoKICysttnkwO9zevBkB / 5
    sc + LY8cKDohKaXUwFm7Fq6 / Hm65BQ4 / HC6 + GGbMGO5RtdOgrTA6Ryq1kTPGBo2dA7n81Ny8fmpp6b6 + sbE9NTTYoDA / iMsv59dVV9u1 + NXVHcujRunVwBFAgzallNoErVsHN99sA7hZs + DSS2HH9e5AHnoatBVG50ilVI + MsVf5GhpsygVynfP6eqirs + v1O + f19fYKXn4gV1Vlb8iuqYHx49vzXLm0dLg / +aeOBm1KKbUJa2qCW2 + Fa6 + FPfe0wdseewzfeDRoK4zOkUqpQRdFNrDrHNCtWQOrVsGKFTatXNle9rz1A7lcefPNbZo6dWBvEv + U06BNKaUUra1w + +3
    ws5 / BdtvB978P + +479
    OPQoK0wOkcqpUYcY + wVvK6CuWXL4OOPYeFCWL4cJkyALbawQVznfNw4fdpnHg3alFJKtclkYM4c + MlPYNo0e + Vt9uyhmzc1aCuMzpFKqY2W78PixTaA + +ij9fN0GjbbzD41a6edYLfdbJo0aZMM5jRoU0optR7fh / vugx // GKZMgcces / eyDzYN2gqjc6RS6lOrocEGcB98AK + 9
    Bi + / bBPY4G333TepQE6DNqWUUt0KQ9h7b / je9 + Doowf / fBq0tRORQ4FfAg5wuzHmZ1300TlSKbXpMAaWLm0P4DoHcrlgbvfdbSD3KaJBm1JKqR7dfz / 88
    pfwzDODfy4N2iwRcYAFwOeAZcCLwFeMMe926qdzpFJq02aMfXnpSy + 1
    B3HPPw9f / zpcffXQLBMZAhq0KaWU6lEYwvTp9l63wX44iQZtlojMBC4zxhwWb18EmM5X23SOVEqpLtTWwpe + ZF9LcPfd9uXjG7n + zI + fjnBVKaVUn7guXHghXHPNcI9kkzIJWJK3vTSuU0op1Zvqapg7F8rK7MtIV6wY7hENCw3alFJqE3PaafDss / Duu712VUoppYZfKgV33QVHHgkzZ8Jbbw33iIacN9wDUEopNbRKSuDcc + 1L
    uH / 96 + EezSbhE2Bq3vbkuG49l19 + eVt51qxZzJo1azDHpZRSGw8R + MEP7LvfZs + Ge + 6
    Bz39 + uEfVJ / PmzWPevHkbdAy9p00ppTZBa9bYe9veeQfGjx + cc + g9bZaIuMB72AeRLAdeAL5qjJnfqZ / OkUop1RdPPQUnnABXXQVnnDHcoymYPohEKaVUn517LlRW2ve3DQYN2trFj / y / nvZH / v + 0
    iz7moDkH4YjTv0Tf + rmOu36ddFHXqV9 + n / y2XH1PdV1td9XWudxTXedjueIin / J3Oyml8ixYAEccAccdBz / 60
    Ub1ZEkN2pRSSvXZhx / CXnvZ952Wlw / 88
    TVoK4yImIn73YjjRohjk + NGiJsrGxzHbudyccCJ22y9QRxj + 7
    rGHsc1tr5DH + J9THxs7H7x / uIacCIcx9aTO65j6yVXLzYZCTFEcbLl0IREJi + P2rfzy7m2zvW5us7tPfXrHMR1zj3H61DnOV6fy7l9PcdrS7n6zuWu + nqOR8JJrFe3Xh830aF / bruQsiMbzz9eldoga9bYJ0tOmGDveSsuHu4R9YkGbUoppQpywgmwzz5w / vkDf2wN2gojIuZrX5tLEBjC0BAE5OW5ZPLKEIbSYTuKcnVCFElcJ20pDAVjJK / OiXOXKBKMcYgiB2Pykxun9m3wMMbB3hrvAW4X5QAIEIkQCYEQEZscJ5dHiEQ4ji3nkhsHl64bxcm0Jc8D1zUkEgbXBc + DRAK8hCGRiHCTEYlkhJeMSCTBS0a2LWXa86QhkTS2PWXzRMomL2FIJAUvBV4CvKRBnBCcAHFDcEKMBIRRQBAFhCYkiMtBFBBGnbZNiB / 6
    Herakmkv5 / r4kU8YhfiR36Gur2U / 8
    nHFJeEm2oK4rvKkmyTh2rxzyrX3lFJuyuZeqttyrl + uvsgrIuXFuZvSK6NqYGQycPrp8NFH8PDDMG7ccI + oVxq0KaWUKsiLL8Kxx9qrbonEwB5bg7bCbKxzZBRFhGFIGIYEQUAQBPh + QGurTzYbksnkp4BsNiKdDshkQrLZqC235aitbJMhm43w / YhsFrLZiCCAbNbg + wbfB9 + HIMjlBt8XgsAGrEEgBIETbwth6LSlKHIIQzcOXF2iyCOKvDhA9doSJIFEXkpig9IsIgGO48d5gOOEuG4QB54hnhfiuhGeZ1MiEZFIGJLJXBKSSftgPJscUimhuNihpMRty0tLPUpLPcrKPEpLE5SVJSgvT1JenqSoSEiloKjIpmTSYKRjENdVng2zHcpdJT9avy0TZNrLYaYtz9W31QUd2zJhhnSQJhPYPBtmSbiJtgCuc0BX5BVRnCi2uVdMcaLY5nF5vfo4L0mUtKXSZGl7OVFKkVekgeKnlTFw + eXwu9 / BI4 / AjBnDPaIeadCmlFKqYAceaO / jPumkgT2uBm2F0TlyZIqiCN / 3
    yWazbXk6naW5OUtLS0BLix / nAa2tuTwknQ5paQnJZCJaW0NaWyPSaZsyGchkDNmsvUhgA1LwfYds1gadvu / g + y5BYFMYuoRhgijyCMMExiQwJgmkECkCioAUxqSAEMfJ4ro + juPjeT6eF + B5IYmETTZojCgqMhQV2VVlJSVCaanTlsrLXcrKElRU2FRZmaSqqoiKCo + SEjqkZNI + 3
    K + vjDFkw6wN5OKALj + oSwdpWoNWm / uttAattPqtbfW5ug7tQSstfktbas4229y3eSbIUJwobgvicgFdWbKMsmQZ5alyyhJ55Vx9srxjn2QZFakKKlOVlKfKdTnqSDJnDnz3u / D739vJbYTSoE0ppVTB / vY3uPhieO21wv7R1RsN2gqjc6QqlDGGbDZLJpMhnU6TTqdpbU3T1JShvj5DY2OWxkafxkafpqaAxkaf5uaQlpaIlhabNzcbWlsNra3Q2gqZjJDJOGQyNmj0fQ / f9wiCBGGYIAyLgGIcpxQoAUowphhjXFw3g + tmSSSyJBI + yWRIKhVQVBRRUhJRWgplZUJ5uUNlpUNFhUd1dZKqqiSjR6eork5RWenG / ey9tuXl9irkQIhMtF5Q1 + w305xtpinbRFO2icZsY3s5E5f9jtuN2UYaM43UZ + ppyjZRmiilsqiSylRlh7wiWdFhu6qoiuriaqqLqxldMprq4mpGFY3SoG + gzZsHX / 4
    y / PGPsO + +wz2aLmnQppRSqmDGwI472ve2HXzwwB1Xg7bC6BypNhbZbJaWlhaam5tpbm6mpaWF + vpm1q3LUFdnU0NDwLp1Po2NIQ0NIU1NEY2NhuZmQ3OzQ2urQzrtkcl4ZLMJfD9FGKYQqcBxyhEpw5gyoqgMgEQiTSqVoajIp7g4oLQ0orzcUF4uVFY6VFV5jB6doKYmxfjxJYwe7VJZCaNG0ZYXFQ3sD1NgA8FcAFefrm / LGzIN69Wty6xjbctaaltrqW2tZW3rWhozjVQWVdpArnh0e1AXl8eUjKGmrIbxZeOpKa2hpqyG8mS5LvPszY9 / bB9S8otfDPdIuqRBm1JKqX656y64 + 274
    xz8G7pgatBVG50i1qYuiiNbWVpqammhsbKShoYGGhgbWrm1i1apWVq9Os2ZNhtragHXrQhoaIhoaoKlJaG52aW1NkE4nyGZLcJwqXHc0UIUx5YRhOeCQSrVSXJylrCygrCxi1CgYPdph / PgEkyYVMWFCEWPHOlRXQ3U1jB5t86KiwfnMYRRSl65rD + TioG5tq81XN69mZfNKVjavZEXTClY2rSQ0YYcgbnzpeGrKatq2J5ZPZGrlVCaUTcB13MEZ + Ej37LP2vTavvjrcI + mSBm1KKaX6JZuFLbaAP / 8
    Zdt11YI6pQVthdI5UamAYY2hsbKSuro5169ZRV1dHXV0dq1c3snx5CytXplm92qe2NqSuzlBXJzQ0eDQ1JfH9ChKJGjxvLFBNGFbh + +U4TkRZWYbycp + qKsO4ccKkSUk226yYCRNcxo2jQ6qoGPirejlN2SZWNtlAbmVTHMzlys0rWNa4jMX1i6ltrWVi + USmVU5jauVUplVOY9qo9vKUyimUJEoGZ5DDzfdtxP3xxzbqHmE0aFNKKdVvP / 85
    vPIK3HvvwBxPg7bC6Byp1PDLZrPU1tayZs2atrR69RqWL69n6dJWVqzIsmpVyJo1UFvr0dRUQio1hWRyEiITiKLRZDKVRJFHZWWG0aMjamqEyZOTTJmSYPJkYdIkmDQJJk + Gmhr72orBkAkyLGlYwuL6xSxat8jm9e35kvolVKQqmDZqGpuN2owZY2aw7dht2XbstkwfPZ0ib5AuLw6VQw + Fs8 + 273
    EbYTRoU0op1W8NDbD55vDyy7DZZht + PA3aCqNzpFIbnzAMWb16NStWrGDFihUsX76cFStWsHTpWhYtauaTTwJWrYK1az2iqIaysukkEpsRRRPJZEbT0lJCVVXAxImw2WYJpkxx2oK6SZNg6lSYNs0 + nXOgRSZiVfMqFtcv5qO6j5i / ej7vrHmHd1a / w0d1HzGlYkpbEJcL6LYZsw2lydKBH8xg + NnPYNkyuP764R7JejRoU0optUH + +78
    hnR6YOU6DtsLoHKnUp1t9fT1Llixh0aJFLF68mMWLF / Pxx5 / w4YfNLFoUsHZtEWVln6G0dDqJxDSiaCItLWNpaChj3LiI6dNdtthC2GILOqTRowd + KaYf + nxQ + wHvrLZBXC6YW7B2AePLxrPt2G3Zbux27DNlH / abuh9jSsYM7AAGwvPPw5lnwhtvDPdI1qNBm1JKqQ2ybBlsvz28 / 779
    h8CG0KCtMDpHKrVpC4KAZcuWtQV0ixYt4qOPPuLddz9k / vwmmpvHMWbMXpSV7YjIVrS2jmfNmkpEOgZzW21l / 45
    vv719auaAjjEKWFi3kPlr5vP6itd5Zskz / Hvpv5lcMZnPTv2sTdM + y9TKqQN74n4NNrAT2QcfwNixwz2aDjRoU0optcFOP91O / JdeumHH0aCtMDpHKqV6sm7dOt5 // 33
    ee + 893
    nvvPRYsWBDnqykp2Z6ampmUlu6EyGdobJzC4sXlVFVJWwC3ww42nzHDvkx9oARRwBsr3 + DpRU / z1OKneHrR0xQnijsEcTPGzBie1xQccYSd1I49dujP3QMN2pRSSm2wd96B2bPtQ7c25DHXGrQVRudIpVR / RFHEJ5980hbIvfXWW7zyyiu88cZbjB8 / k6lTD6e0dE8ymemsWDGaDz90mTKlYyC3444wffrALLM0xrBg7QKeXvy0TYuepjHbyH5T92P / qftz / HbHM7li8oafqC9 + / nM7md1009Ccr480aFNKKTUgjjwSvvhFOOus / h9Dg7bC6ByplBpIvu / zzjvv8PLLL7elN998k8mTN2f69MOpqjoAke2prZ3Im28maGqCffe1ab / 9
    YLfdIJUamLEsbVjKvxb / i8c / epwH5z / IXpP34vSdT + eLn / kiKW + ATtKVl16Cr30N3n578M7RDxq0KaWUGhBPPQVnnAHz54Pbz3ezatBWGJ0jlVKDzfd95s + fz8svv8xLL73UFshtttlm7L33cYwdezT19dvz / PMJ3nsPdtnFBnD77Qf77ANVVRs + hha / hYfmP8Qdr93B6yte58QdTuT0XU5n5 / E7b / jBOwtDGDMG3n3Xvl9hhNCgTSml1IAwBmbOhIsu6v8rbjRoK4zOkUqp4RAEAa + 99
    hpz587lscce45VXXmHmzJkccMCR1NR8gaVLN + eZZ4QXXrCvIMhdifvsZzf89TAL6xby29d + y29f / y2ji0dz + i6nc + IOJ1JdPIAvxP7iF + Hkk + GEEwbxwYzHAAAgAElEQVTumBtIgzallFID5sEH7e0Azz7bv / scNGgrjM6RSqmRoKGhgSeeeIK // / 3
    vPPbYY2QyGQ4 + +GAOOuhQxo8 / hLffHsW // gVPPgkTJsDxx9s0fXr / zxmZiH8u / Cd3vHoHf3v / bxy61aF8feevc9AWB + E6 / VzukXPddbBgAfzP / 2
    zYcQaQBm1KKaUGTBjCNtvAnXfaX1ULpUFbYXSOVEqNNMYYPvjgAx577DEee + wxnnzySWbMmMEhhxzCwQcfRhjO5IEHhAcftKsQcwHcNtv0 / 5
    x1rXXc99Z93PHqHaxsXslpO53GBXtfQFVxP9dmvvoqfPWrdonkCKFBm1JKqQF1663w6KPw8MOF76tBW2FExKx + 4
    yXEcRHHQRwXxMFxHHCdDvXiuu3bbnu943ngODiua3PHGe6PpZT6FMlkMjz77LM89thjPPzww4gI55xzDieddApvvVXJ / ffbVRpVVe0B3Lbb9v98r694nRtfuJG5H87lrqPv4sDNDyz8IFFkI8q33oKJE / s / mAGkQZtSSqkB1dpq57j582H8 + ML21aCtMCJi5t27GUgUJwMSYfLKiAEnzMvpoj7ediN74NCByAEjNo8cMLlt1 + amY70YN6 / dAWxZyK9z2vvhdirb9vayi9Bb7iLi2rLkbYsTl7327bhsA1sXx2nfFifRXnY929dxcdxEW47rtG97CcRN4Lgu4iVwvASO4yKJRNxm + ziJ9iSupwGx2uQZY3jqqae45ZZbmDt3LieccALnnHMOO + ywE // +N9x / PzzwAFRW2uDtuONgu + 36
    t9z + 0
    fcf5Rt // gan7HgKV86 + kqSbLOwAX / qSvaftq18t / OSDQIM2pZRSA + 7
    oo + ErX7GpEBq0FWYw5sgoDDFRiAkjmwchURRCFBEFga2LIkwY2DzuY0ynujDuFwVEYQCRwURBW13bcaIQY + JzxmUie04ThWAijAna + 5
    n2fTC5uiDejtq3ydsmjLfjnLy6TrmREAhs8JvLJQQJ23IkxDhx2bGpw7Yb2kDYDdq33cgGw6Frg9jOeeQikRcHuLmyhxhbJyZh600CG9gmbGBKAsEDPFsWu + 1
    IXBYPcRI4krTBqZNAHA / HTeC4KRwngeOlbBCaSOJ6KRwviZOI82QSN5HCSSZte7IIr6gYJ5WyAakGoqqfli9fzu23385tt93G1KlTOeecczjuuONIJFI8 / 3
    x7AFdWBt / 5
    Dpx6KnheYedY1byKM / 58
    Bp80fsI9x9zDNmMKWIN5ww32StuvflXYSQeJBm1KKaUG3PXX21fcFDrXadBWGJ0jNx5RHOhGvt + WTBgSBT5R4GN8nygMCP0MJgji + qzdJ4z7hAFRmG3LoyjARD4m9IkiHxP5RFHW1kUBxvhEJhsHtXbbkEsBhqzNxQfxMbnk + CABxvXBCcD1MXFOwrfBqOeDF0I2AUGcQg / CBBIkIUogYcIGmVESiRKISdlEEockQgpHUogkcSSF4xThOCnETeF6RTheEa5XjJMoxkuW4CaLcVPFeKli3KIS3KJivOJSvOISvOJiu9RXbXSCIOCvf / 0
    rt9xyC6 + 99
    hqnn346Z599NptvvjlRZF8nc8UVsHQpXHaZvfBVyGtljDH86uVfcck / L + Gq2Vdx9m5nI325dPfGG3DssfD + + / 3 / cANIgzallFID7s037cqSDz4obD8N2gqjc6QaTlEYEmbShJkMYSZNlM0QpNNEfoYwm4nzNJGfJvTj + iBNFKSJwkxeShOZDCbKEpkMkUljyBCRwZDBSAbj5CUvDV4W42XAy0IiC8msvWqZTUG2CAlS4BchYRESppCoGCcqQijGoQiHEhynGNcpwXGLcb0yvEQZbrIUL1WGV1RGoqQCr6Qcr7SMZGkFifIKvKKi4f7aP9UWLFjArbfeypw5c5g5cybf + ta3OPTQQ3FdlyeegO9 / H2pr4fLL7dLJQi70vrvmXU7640lMKp / Eb774G8aVjut5hyiy72l79VWYPHmDPtdA0KBNKaXUgIsiez / bSy / Zd / T0lQZthdE5UikriiKiTAa / uYmgpYmgtYWgtZmgtZkw00KQaSLMthD4zYR + K1HQQhi2EIWthFEzkWkhIk5OC8ZpwXitNiVbIdkKqbS99zNTDJliJFOK + KU4YSlOWIaDTa5ThudW4HrleMlyvFQFieJRJMuqSFZUk6ysoqh6DImS0uH + 2
    kaslpYW / vCHP3DzzTdTW1vLjTfeyBFHHIExMHeuDd7SaXsF7uij + 37
    PWzbM8oMnfsCc1 + dw + xdv57CtD + t5h + OOsyc4 + eQN / 1
    AbSIM2pZRSg + KEE + CII + BrX + v7Phq0FUbnSKWGVpDJ4DfUk22qx2 + sx29uINuyjqClniDTiJ9tIPQbCMMmgqiBiCabnCYirxGTasSkmqC00T6Ep6UcSZcj2XKcoBw3qsClEtcZhedW4qWqSZWMJVU + ltSoGorH1FA8dvwmdcXv8ccf54wzzmD27Nlcd911VFRUYAw88gj84Ae2zw9 / aOebvgZv8z6ex6kPncpRnzmKqz9 / NcWJ4q473nwzvPIK3H77wHyYDaBBm1JKqUFx663w73 / DXXf1fR8N2gqjc6RSG6coighaW8jUriGzbi3ZhlqyTbVkW9fhp + sIsnUEwTqCqI6QWkKvjihVhymuh7J6yKaQ5kokXYXjj8KNqvFkNAm3mkTROIrLJ1FcPZmSmkmUTpy60Qd5jY2NXHjhhcydO5c77riD2bNnA2AMPPSQvdettNQGb5 // fN + Ct7rWOr71yLd4c9Wb3HvMvew0fqf1O739NnzhC / DRRwP8iQqnQZtSSqlBsWABHHQQLFrU918 / NWgrjM6RSm16oigiu66O1tUrSNeuIl2 / kmzzarKta / D9NfjhagJZSZhcTVS6BsrroLUUaRqNmx6LF9WQkBqSqQmkyiZSPGoipTXTqNhsa7zibq44jRCPPvooZ555Jscccww // elPKSkpAeyS / Pvvt / e6jRkDV14Js2b1fjxjDHe / cTcXzL2Ai / e7mPNnno8jTn4Hu9b / hRdg2rRB + Ux9NWhBm4gcCvwS + 9
    KV240xP + vUXgHcDUwFXOBaY8xvuziOTkhKKbURMsbeu / 3
    kk7DVVn3bR4O2wugcqZTqTRSGtKxcTsuKpbSuWUprw1IyzcvI + isJWEngrSQqXYmpXANNlbiNk / D8KaTcqRSXbk7J6C0on7S1DepGwBW72tpazjvvPF544QV + +9
    vfss8 + +7
    S1hSHcd59dNvmlL8E11 / TtYSUL6xZy3P3H8bWdvsZ5e53XsfHLX4bDDy9srf8gGJSgTUQcYAHwOWAZ8CLwFWPMu3l9LgYqjDEXi8gY4D2gxhgTdDqWTkhKKbWROuUU + Oxn4ayz + tZfg7bC6ByplBoooZ + lacnHNC55n + Y1H9LauJBMsAjfXUJY9glm1GpoqMZtnEgimELK3Yyy6h2o2nwXRn1mB7xUakjH + +CDD3Luuedy2mmnccUVV5DKO39tLRx1FEyZAr / 9L
    ST78F7tFz95kePuP44P / uMDEm6iveHWW + G55 + yBhtFgBW0zgcuMMYfF2xcBJv9qW1w32RjzbRHZHHjMGDO9i2PphKSUUhupO + +0
    T / q6776 + 9
    degrTA6RyqlhkroZ2n8 + CMaP7FBXUvj + 6
    TD9 / DLF2BGrUTWTiHZMp2SxHaUj9mRqq12pXKrbXAKealagVatWsU3v / lN3n // fe666y523XXXtrbWVjjpJGhogD / +ESoqej / egXcdyBm7nMFJO57UXvnuu3DIIfDxx31f6z8IBitoOxY4xBhzVrx9MrCnMea8vD5lwJ + BbYAy4MvGmEe7OJZOSEoptZH6 + GPYay9YsaJvc50GbYXROVIpNRJkGxupe / c11i1 + lab6t0mb + fiVC6CsHmf15iQzn6EktS2VNbtQs8dsiqqqB + zcxhjuueceLrjgAr797W9z8cUXk0jYK2VhCP / xH / ahWH / 7
    G0yY0POxHn3 / US5 + / GJePfvV9hdwGwMTJ8Izz8AWWwzYuAs1nEHbscA + xpgLRWRL4B / AjsaYpk7H0glJKaU2YltsAX / 5
    C2y3Xe99NWgDEbka + AKQAT4Evm6Maeimr86RSqkRK127ltr5r1D / yes0Nb5F2n2LcNw7OGs3oyS7N9UTZjNhz4MpHjN2g8 + 1
    dOlSzjjjDNasWcOcOXPYdtttARtz / fjH9qn9f / 87
    TF9vXV87Yww73roj1x58LQdveXB7w4kn2idrnX76Bo + zv / ozP3p96PMJ9gEjOZPjunxfB34CYIz5UEQWYq + 6
    vdT5YJdffnlbedasWczqy + NglFJKjQizZ8MTT3QdtM2bN4958 + YN + ZhGuLnARcaYSER + ClwcJ6WU2qgUVY9m4r6fZyKfb6sLWltZ8eITrFn4OMtW3MjiF87EqZ1CcWZvqmtmMWGPQympGV / wuSZPnsyjjz7Kb37zGw444ADuvvtuDjnkEETgkkvsVbb994eHH7YrQLoiIvzXPv / F1c9c3TFomzXLTmTDGLT1R1 + utLnYB4t8DlgOvAB81RgzP6 / PzcAqY8wVIlKDDdZ2MsbUdjqW / oqolFIbsXvugQcftPcU9EavtHUkIkcDxxpjTummXedIpdRGLUinWfXy06z + 8
    P9oCp7BH / 8
    qsm4CJS17UzVuFuN3P4SyiZMLOubTTz / NcccdxzPPPMNWeY8vfuQR + PrX7f3WRxzR9b5 + 6L
    PlDVvy0JcfYreJu9nK99 + HAw + EJUuG7b62wX7k // W0P / L / pyJyNvaBJL8SkQnAb4Hc6tKfGGPWu1VdJySllNq4LVsG228Pa9b0 / uhlDdo6EpE / A783xtzbTbvOkUqpT5XQz7LqpWdY / cE / aMz + C7 / mVbxV21FTfTqbH3JKn98ld + utt3LjjTfy3HPPUV5e3lb // PP2yZI // nH3F86u + / d1PP / J8 / z + uN / bCmPsoyifeAK23npDP2K / 6
    Mu1lVJKDboZM + Dee2GXXXrut6kEbSLyD6AmvwowwCXGmL / EfS4BdjXGHNvDcXSOVEp9qvktzSycezerau8kGPMepau + yLRdv8W4XWf2uu / ZZ5 / NypUr + eMf / 4
    iT96vhe + / BYYfZoO2SS9a / eNaYaWTz6zfnhTNfYIuq + OEjp5xi11eeeeZAfrw + 06
    BNKaXUoDvnHNhyS7jwwp77bSpBW29E5DTgTGC2MSbTQz9z2WWXtW3rfd9KqU + zugXz + fiZ / 6
    G + 4
    g84zWMYkzyVLQ45s9unUWazWWbPns1BBx3U4RkZAMuX23dm77033HgjdH4zwSWPX0J9pp6bDr / JVtx + Ozz + uP0Fcgh0vuf7iiuu0KBNKaXU4HrgAXsPwSOP9NxPg7a22wuuBfY3xqztpa / OkUqpTU4UBCx + / EGWLbmd7IR / U7T8ICZPP5OJ + x3a4YoawIoVK9hjjz248cYbOfroozu0NTTAMcfYd7jdcw / kr7xc0bSCbW / elve + / R5jS8fCRx / BvvvaNf / DcF + bXmlTSik16NassVfa1qyB + PU5XdKgDUTkfSAJ5AK254wx53TTV + dIpdQmrWnZUhY + fhu13t0gEVXZk9h89tmUT57W1ufFF1 / k8MMPZ968eWzX6VHG2Sycdpp9xshf / gKjRrW3nf2Xs5lQPoHLZ11u72vbbDN47DHYZpsh + Wz5NGhTSik1JHbaCW67DWb2cBuCBm2F0TlSKaWsKIpY / uz / sfTd22gdP5fK2tPY8cRrcb0kAHPmzOHKK6 / khRdeoKqqqtO + 9
    v62igq44Yb2 + gVrF7DfHfux8P8tpDRZaqO7mTPhm98cwk9m9Wd + 7
    OXZX0oppdT6Zs + Gf / 5
    zuEehlFLq08hxHCbtdzB7nfEgu + 3
    wFs3pV3j2f3ejYZl949ipp57KkUceyVe + 8
    hXCMOy0L1x9tV0iuWhRe / 300
    dP57LTPcudrd9qK3PvaNhIatCmllCpY7iXbSiml1GAqnzaNvU97ivK6Y3jlpX346NmbMcZwzTXXEIYhF1988Xr7jBsH3 / oW / PCHHev / a5 // 4
    tp / X0sQBfZdbfPm2aWSGwEN2pRSShVs // 3
    huecg0 + 2
    zEJVSSqmB4SZddj73CrY0D7P4 / et4 + c9HEkXr + MMf / sADDzzAffet93povvMd + POf7SsBcvaavBdTK6fywDsPwLRpUFoK8 + cP4SfpPw3alFJKFayy0r6v7bnnhnskSimlNhVTjtqf3fd5gfSrlTw3dwdM5mX + 9
    Kc / cd555 / Hqq6926DtqFFxwAeS9SQWwV9uufuZqjDH2attGsmxEgzallFL9oksklVJKDbWyravZ + 79 / R + XzV / H28yeTTN / BTTddz5e + 9
    CVWrVrVoe9558GTT8Jrr7XXHbb1YWTDLI8vfHyjuq9NgzallFL9cuCB + jASpZRSQ88tdtnpym + wRcM / WTnvFaZW / 4
    Tjjz + E448 / Ht / 32 / qVlsLFF8Oll7bv64jDd / f5Llc / c3X7fW1RNPQfokAatCmllOqX / faDV16BlpbhHolSSqlN0ZSvbc + uBz9CdN + RHDH7DyS8tZx // vkd + px9Nrz5Jjz7bHvdV3f4KvPXzOdVdzVUVcHbbw / xyAunQZtSSql + KS2FXXaBZ54Z7pEopZTaVJXvXM5e111J1QN3c8EJPn975C5uu + 26
    tvZUyt7Xdskl7Q + KTLpJzt / rfK559pqN5r42DdqUUkr1my6RVEopNdy8So8df3ME27f + jSv3 / xwXXXQhTz55T1v7qafCsmXw + OPt + 5
    y525nM / XAuq / bYVoM2pZRSn276km2llFIjgYgw9fwt + cLZc / jWNodzzjfPIgwDADzPvrPte99rv9pWkargjF3P4IaSN + 3
    TSkb4fW0atCmllOq3mTPhnXegvn64R6KUUkpB5b6VnPvNWzG + 4
    ZZbvt1Wf / zxkM3Cww + 39
    z1vr / O4ZelDBGNHwxtvDMNo + 06
    DNqWUUv1WVAR77glPPz3cI1FKKaWsmmPH8 + 3
    kufzwh79m3boVADgOXHUVfP / 7
    EIa238TyiRwz4xhe22bUiF8iqUGbUkqpDaJLJJVSSo0kXpnHgXuczu7bTuKSS45rqz / iCCgvh9 // vr3vd / b5Dr8qX0Dw + P8Nw0j7ToM2pZRSG0Rfsq2UUmqkGX / qeM7JXsnddz / L / PlPASACP / qRfZpk7nVu24zZhsx + exM89UT7JbgRSIM2pZRSG2T33eHDD2Ht2uEeiVJKKWWNmjWKMUunc9pX9uf8809sqz / wQNh8c7jzzva + Zx / xAxaXBgSvvDQMI + 0
    bDdqUUkptkETCvmh73rzhHolSSilliSvUnFzDGWXX8sYbK3jkkevb2n70I7jySkin7fY + U / bhrRmjefv + W4ZptL3ToE0ppdQG0yWSSimlRpqaU2povC / DFVd8m // 8
    z4sIgixgH6C1227wP // T3rfswEPw // XUMI20dxq0KaWU2mD6km2llFIjTem2paQmpThm2qWUlSX5xS9Ob2u78kr46U + hsdFuT / z8MUx5Z2n7i9xGGDFDODARMUN5PqWUUkMjDGHsWHj7bZgwwdaJCMYYGd6RbTx0jlRKqYG39MalNDzfQO03X + eoo07m3Xc / YuzYaQCcdBJss419DYAfZKkbVUTRK29QMX37QR1Tf + ZHvdKmlFJqg7kuHHCA3temlFJqZBn3lXGs / etaZu50PJ / 73
    NZcdNGxbW1XXAHXXw + 1
    tZDwkizYupqP // 77
    Ho42fPRKm1JKqQFxww3w5pvw61 / bbb3SVhgRMf9YuxZHBAcGLXdFuq6Lt8WOZei / AKWUGiRvHvUmY740huBza9lhhx2ZN + 8
    Rdt75MADOOguqq + 1
    SyUe / cQBVDT4z7392UMfTn / lRgzallFID4q234Kij7OP / QYO2QomI + dyrrxIBkTG95iFg + tAvPw + 7
    Ok6nOgMIHQO5zoFdh3If6tx + ll3ag0w3v77TObpqy5W9Hvp4edteL / VeXpvXqb5zuattBw2GlRouqx9czSc3fcLOT + zMJZccydNPv8hTT60EYMkS2Gknu7z / g7lXMuaqXzDj / bpBHY8GbUoppYaNMTB + PLzwAkybpkFboUTEvP23u3DcBOK6OF4Cx7Nl10vasuPiJmzZcT0cL4HrJW2fRBLXS9p217P1TuF3QXQVCIadgsUOQWAvdWEB5fx9w7zz5so99Qu72Q7yt7voE + TVBT3Ud24L8voEnfp0tR3RMeDLpUQXdZ3bEiIkHGe9uq7aEnl1XW0nO7UlHcfW5ZfjvKu2XO6KaBCqNhpRJuLZic + y + yu7Q03A1ltXc + 21
    P + CEEy4F4IIL7Mu2L7nkQ8qnbkVJQytSVDRo49GgTSml1LD68pfhsMPgtNM0aCuUiJh3phYjBpzQ4EQGMTZ3IoNjWK / sGnAicIzBjbDbhrZyBIQORGLzUCByIHSESCASse2OxCm / nJ8cItfmxhEiN5e7GEcwroNxbIo8F + M64DgYNy67LsZzwYlz1wXPs3l + 2
    fMQLxHntiyJRHvuekgigeMlbZ5IdkhuImXLSVt2Eym8omJbTsbbySLcpM0TqRIc1 + tXcFuoKC + A8zsFdvnJj / v5xuBHUYf + +XlXbbl6v4vtbC9t2Vw5ijpud2rLxLkBUnEAl8oL6LqrS4lQlCs7Tns5r75ze5HjUNxd7rptfVwNHlUfLPjWAlKTU0y7ZBr33PN9Lr30at59t45UqoRVq2DGDHj5ZWjaL0X5b + Yw7dAvD9pYNGhTSik1rG67DZ55BubM0aCtUAM9R5ooIgoDojAgDLKEfpYwyBL5vs3DgCjwiQKf0M8ShT6R73edBz4mCIj8rC3H + 5
    ogIAqymCCwKYzzuI0w7FQOIAwg3m7Lc + UgROJtCQIkjJDQ1kkQImGEE4ZxfYQTRrhBiETGlsPIBrNhhBMavDDCDQ1uZHBDcCODFxm8ELzIJtdA1oXAAd + B0BUCRwhdCFyH0BWbHIfQE0LXIfRcIi / OXZco4WJyuefZYDXhxSlh30CfTNo8kYBkCkkmkGQKSSRxUkVIMmWDzVQxTqoIN1WMW1SMV1TSlieKSkmUlNm8uJRkcRnJ4jIc1xuw / 9 / 0
    VZgL4uIgL5MX7HWuy8R1mSgiHQd + 6
    c51eW25cjqKaM0vh2GHulzuxYFfseNQ4rqUxHmx47SVe8vL4lSayx2nra7EdXE0MNzo1T9Xz7tfe5c9390TYwz77juWww6bxQ9 + 8
    CAAF14IxcWw / 4
    tbUbXDnuzx83sHbSz9mR + H / r9ypZRSn1qzZ8NVV43Y19xsUsRxcB27XDJByXAPZ8SKwgCyaUy6xeaZVvAzmHScZ1oxfgaTSWOyaUwmDVm7jZ + BTMb2y2Ygm8VkMxjfBrK2LWvXXfk + tLZCQwOS9W2Q6geIb8viB0gQgh9CEKe4LEGE + BESRBAaCIzNQ3ACG3BmXci6QtYTfE / wEw6 + 5
    xAkHIKES + i5BEmPMOERJT2ihEeUTGJSSaJUClJJKCqCVAopKkZSRUhxMW5xKU6cu0UleCVlJErLSZRVkCobRbK0gqLyKirLq0gWlw3JVct8Jr4amAvqWqOIliiiJQxpievyt9vqo4i6TIbmeLspDGkKQ5rjPL / cEkUUO05bQFfmupS7LhWeR0Wnck95ZVzWZaXDo2KvCjDQ + EIjFXtVcMMNd3DwwV / ijDPeY + LEz3DIIXb + +uyBM5F / Pjfcw12PXmlTSik1YIyBKVPgiSdg + nS90lYInSNVf5goIgyyZJobyLY04qebyTY34rc2EaRbCFqb21KYaSVMtxC1thJl0kStLUSZNKa1FTJpSKchk0HSGSSTQbI + biaLk / VxswFe1sfLBnjZkGQ2JJUNSfqGlB9RFEAihHQCMp6QTgjZpEM24eInXbJFCfxUgrAoGaciTHERpqQYSkqgpBSnuASntAy3rByvrIJExSiS5VUkK6soHjWW4soxlFSNJVVSMaTBYWQMrXmBXS41BAENfczrg4D6MKQ1DKn0PKo8j6pEwuY9bFd7HuOSScYmEiSHOCD + NPr4qo / JLs8y / ebpAJxyyo5EkeGee96ksdHel / 3
    En + 9
    nyvEnMaE2O2jj0CttSimlhpWIvdr2z38O90iU2jSI4 + Ali / CSRZRWjRvWsYR + FtO0DtNUD03roLkh3m6ApgZoqsc0NWKaG6G5EdPcDK0t0NICtXX2SmRrBtIZSGdtavVx0gGSDZFMhGQNoYHmBLSmHNIpl3SRRzblkS1O4pcUEZQWE5WVYMrLobwcqajErRhForKKRGU1RdXjKKoeS3HVOEpHj6esenyPS0wdEUrjpZM1G / gd + VHEuiCgLpd8v70cBKzKZnmvpaVte63vs9r3WeP7lDgOYxOJtiBubCLB2GSScXnlsYkENckkNYkEngZ566k5uYaXd3 + ZrX6xFU7K4ec / f4htttma559 / kL32OpYZM6DR / QKJjE / zwgWUbj59uIfcRoM2pZRSA2r2bPj734d7FEqpoeYmkpRWjRv04NFPt8C61Zh1q4nWrcbU12Ia6jD1tdCwDtbVQkM9NDbA2rXIosVIUzM0p5GWDLRkkbSP0xriZCKMD + tS / 7 + 9 + 47
    Pqrz7OP75Ze + dEGZA9ka2zICIggpStXW2tlXUttantba2tYptbbWt3Y + tWrW1j3WL1dYtxgVi2AEiewYCScgO2dfzxx1CAiFk3BmQ7 / v1Oq / cZ13nd6fay2 / OOddlFAX7UBzqz9HQQMrCgqkIC6EqMgwXGQmRkfhGx + IXHUtgXDeC43sQ3qMvEd37Eh7bo8l3 / vx9fDzhKiCgWd / ZOUdeZSVZFRUcLi8nqybMZVVUsKe0lNTCQs96eTmHKirIqagg3t + fnoGBniUgoPZzrzrbwvy6VhQI7htM6IhQcl7PIX5RPN269eeOOy7n29 + +iRUrFjFjhg8rlwcRNCCKyDefYcSt93Z0ybX0eKSIiHjVnj0wYQJkZenxyOZQHynSMSrLSynMyqAoK4PiwxmUHjlEWfYhKo5kU5Wbg8vPg7x8fAsK8S8sJqDoKMGFpYQXlRNZXEVQJbz7oBUAACAASURBVOSFGAVh / hSHBXI0MpTyqHCqoiIgNhaf2Dj84xMJ7t6b8N4DiOk7lKju / dr0Ec + K6moyy8vJKCtjf1kZGTWfa5eadX8zegYG0jswkP7BwQyos5wTFESQr2 + b1dhRDj5xkJzXchixdAQA5eWlDB0axT33fIeIiF / yyCPw7cSpxJT5MvmZD9ukBo0eKSIincKgQbBtm0Jbc6iPFDkzlRUXkJ + 5
    m4KMXRQf2kfpoQzKDx + kKicLcnLwyc0nIK + A4LwiIvKOEl1QSUiFIyfUh7zIQIqiQymLiaAyPha6dcMvsSfBPZMI692fmHOGE9tncJuMEHrs7l1GWRl7y8rYcfQo2 + sse0pL6RYQUC / IHVv6BwcTcoYGusqCSlb0WcGk7ZMIiPPc8XzppQe4 / fafsHz5EUaODOeV3 / yEbg89zLDPc9qkBoU2ERHpFMrLITBQoe0YM7sD + DUQ55w7copj3JgxDl9f8PE5PoVZUz43tH6qbS1dGppW7XTrdX82Z5ufn + f9SJGzVVlxATm708nbu4Xi / bsozdhDVeYB7NBh / HKOEJxTQHheCTF55YSVOQ5F + ZEbG0JRt2gqunfDevUmsG9 / IgYMJ27QGOL6DvN6sKusrmZfWRnbTghz248eZVdpKbF + fowJC2NceDhjw8MZGxZGr8DAM2J0zM3XbCZyaiQ9v9mzdltyciJz5ybzf // 3L
    L / 7
    QzrTLx5GcGEpFhjo9esrtImISKehedo8zKwX8DdgMDCusdC2erWjuvr41GVN + dzQ + qm2tXY5cVq1puyrOyXbidOznbit7mcfn4bDXFMXf // Tbzs2bVpzPzc09dqploCA48fVPf4M + O9a6SSOFhzh8Na15G5Po3jnFir27MQyDhCUmU1EVj5xOaVElDqyInzJiQulKDGain5J + A0aSuTwsXQfM53YPoO9 + jhmlXPsLS1lbVERawoLWVNUxOrCQhwwNiysNsSNDQ + nX1BQpwtyOW / msPve3YxbOa522 + OP38E // vFPhg49zJAhjgt / HUDkP1 + g5 / mXef36Cm0iItJpKLR5mNkLwE + BVzlNaFMf6eGcJ3QeC3InBru6S0VF / X01056ddExD63V / NvVz3aW8vOHtDe0 / 9
    rm83NOen1 / Dge7YtmNLYGDT1wMDG19OPCYo6PhybD0wUIHyTFRalMfhbevI3ZZG4dY0KrdtwX / nHqL2Z9MzswTDkZEQQl6vWMrPScJv4BAiho + l + 9
    gZxCUN9Uqgc85xoLy8NsStKSxkdWEhxdXVtQFuXFgY50dHN3sgFm + rrqzm096fMvr90YQOCQUgM3Mb / fsP4s9 / Psp // hPEzfn9iBk / nfEPPOX16yu0iYhIp6HQBma2AEh2zn3XzHah0CZ4QumJQe7Yz7qfy8qObzvdellZ05djx9dMy0Zp6fHF82hzw4Hu2BIcfPznseV06yEhniU09PjnY4uCYts7sm8bB9Z9SH7aKiq2puO / aw + R + 7L
    ocagEvyrH / m4h5AzsSfXoUURPmU3fGQuIiO / llWsfKi9nbU2Q + 6
    yggJS8PIaEhDA / NpaLY2M5NywMnw74B2D797bjE + jDOfefU7tt6NAQ7rrrr9x555f5x41fIn75OsanbPH6tRXaRESk0 + gqoc3M3oF60zcZ4IC7gR8BFzjnCmtC23jnXINvtpuZu / fe48NLJycnk5yc3GZ1izSkutoT3OoGudLj8257plI76lk / 9
    rmh9brbSkqO / 2
    xoqajwBLsTA11oKISFHf9Z9 / OptoWHQ0SE52doqOcxW2lcbsYO9qe + R + 7
    KFFi7lpit + +i3v5isKH8O9k + gbMRQQidMpXfyAroNGNPqu3Ll1dV8lJ / Pf3Ny + G9ODgVVVcyPieHi2FguiI4mvJ2mISjaUETaJWlM3j0Z8 / F0VYsXTyQ0NJSXX36fP9 // NGP / 52
    v0zC5r9bVSUlJISUmpXb / vvvsU2kREpHPoKqHtVMxsBPAuUIInyPUCMoCJzrnDDRyvPlK6pKqqk0NdcbFnKSryLE35XFQEhYWepaDAExpDQ4 + HuFP9jIryLJGRxz / XXdpgHIpOr7K8lN2fvc2hj9 + iYk0q4ek7SNqViwP29oumcNgAAsZPYsAXbiSh / 6
    hWXWt7SQn / PXKE13NyWF5QwKTw8Nq7cIOCg9v0fbjUMakM + N0AomdFA / Dii / dz // 2 / YsSIfCZPLebq74YRvGUnwb37efW6utMmIiKdRlcPbSequdM21jmXe4r96iNFvKiy8niQKyg49c / 8
    fM + Sl9fw4ut7cpCLjobY2MaX8PCz67FPV11N5tY17P / wPxSnfkLwuo0M3niIrOgADkwcSvCFlzDkiptb9VhlUWUl7 + Xl1d6FC / bx4eLYWG7u0YNhoaFe / DYe + 367j + K0YoY8OQSAgoLDJCZ24xe / yGP16khuXRtJ9O13MfSmH3r1um0W2szsIuD3gA / wuHPuwQaOSQZ + B / gDWc65WQ0cow5JRKSLUGirz8x24nk8Uu + 0
    iZwhnPPcBawb6nJzPUtOzvElO7v + ek6O5zHTmBhPgIuLg / h46Nbt + JKYWH89JKSjv23zVVWUs + XtZzj86r + I + DiVgdtz2d07jJwp5xI1 / 3
    KGXvpVAkMjWtS2c471RUW8kp3NXw4cYFpkJD9OSmJseLjX6i / LLCN1aCrn7T8P31DPvHPjx0dxzTW / 4
    I9 // Ab / O20ScRbKpH8u89o1oY1Cm5n5AFuB84EDQCpwlXPu8zrHRALLgbnOuQwzi3POZTfQljokEZEuQqGtedRHipxdysrqh7isLMjMhEOHji911 / 39
    Tw5y3btD7971l + Dgjv5mp3a04Ajp // 4
    bBa8vJWHFBnpllrB1UCxFMyaTuOBaBp1 / ZYvmkyuuquKxAwf4zb59jAoL48dJSUyNjPRKzRsu3kDC1QkkXpcIwB13zCE3N4 / XXlvFo3fexaC / P87wzVleudYxbRXaJgP3Oufm1azfBbi6d9vM7Fagu3PuntO0pQ5JRKSLUGhrHvWRIl2Xc55HNU8MdQcOwL59x5eMDM + gKycGubpLz56e6R06g9yMHWx56RHK3n6DPqlbCCuuZPP5o + jx3XsZOHNRs9srq67m75mZPLh3L0lBQdydlMTsqKhWvfd2 + LnDHHz8IKPfHg3AO + 88
    yje / eTsjRhzl / DnrueE75xJaXO6Zp8NL2iq0XQ5c6JxbXLN + HZ6XqL9d55hjj0UOB8KAPzrn / tlAW + qQRES6CIW25lEfKSKnU13tuWNXN8jt319 / PTPTE9wGDjx56dvXc0evo + xdm8LO3 / 6
    Ewa8uJzsuhLzrrmDs / zxIaHRCs9qpqK7mmcOH + cWePUT5 + XF3UhIXx8a2KLxVHa1iRc8VTEibQGDPQMrLS4mJCeaOOw6Seagb33nFn6jnXiVx5vxmt30qHRna / gSMA2YDocAKYL5zbvsJbalDEhHpIhTamkd9pIh4Q0UF7NoF27advBw4AH36nBzmBg + GpKT2GzilsryUNY / fD489ysDPs0ibOYT423 / E0Iuua1Y7Vc7xclYW9 + / ZA8CPkpK4PD4e32Z + kS03bSF4YDB9vt8HgOTkRKZOvYNXXrmT3yUkETd1DmN // niz2mxMS / rHptznywD61Fk / NmRxXfuBbOdcKVBqZh8Co4HtJxzHkiVLaj9rDhoRkbPHifPQiIhI + / P3h0GDPMuJyspg587jIW7jRli6FNLTPQOujBkD554LY8d6fg4e7NWnAmv5BQQx8dafwa0 / 42
    B6KtW / uYvwa24gPWwxWVcvZPR3HySyW5 / TtuNrxpUJCVwRH89 / c3L4 + Z493LNrFz9MSuLahAT8mjinXLcvd2PrLVvpfWdvzIxZs6aQlvYP9u + / k + ILxhGzYnlrv3KrNeVOmy + wBc9AJAeBz4CrnXPpdY4ZAvwJuAgIBFYCX3LObT6hLf0VUUSki9CdtuZRHykiHenwYVi79viyZo3nztyIEfWD3MiREBTk / etXV1Wy5qkHqXjkLwxbn8GGKQOI + tb3GLHwpiZP6O2cY1leHj / ZtYtYf3 + eHzaMYF / f059X7Vg5YCXDXxxO + NhwPvtsKYsWfZERIypYNOsJLvntN + h1uLS1X7FWWw / 5 / weOD / n / gJndjGdAkkdrjvke8FWgCnjMOfenBtpRhyQi0kUotDWP + kgR6WwKCmD9 + vpBbutWGDAAxo2DWbNgzhzPO3TelLVzI5t + 832
    SXnyXSj8f8pbcxYTFS5p8fkV1NV / 5 / HMOlJXx6siRRDThduGun + yiqqSKAQ8NoLq6mvh4f667bjvON5af / iWS4J37COze8jno6tLk2iIi0mkotDWP + kgROROUlsKmTZCaCu + 9
    B8uWeaYnmDMHLrgAZs6EiJZNzXYSV13Nqid + Tre7fsbeEb0Z9s + 3
    iOk9sEnnVjnHt7ZtY1VhIW + MHEncaYbULNpQRNqCNCbvmoyZccklfenTZzGpqT / ij8XhxNx5L4O / +j1vfK0W9Y9Nu9coIiIiIiJdXlCQ5y7bLbfACy94Hqt86ino0QP + 8
    AfPXbdp02DJEvjkE8 / AKC1lPj5MuPEeYrdlUBUVSfmIIaz43R1NOtfXjIcHDuSC6GhmrFtHRllZo8eHjgzFJ8CHwtWFAMyePYvt258kPR2yRgwi / 4
    O3W / 5
    FvEB32kREpE3oTlvzqI8UkbNBSYknrL3zDrz7LuzY4bn7duxO3NChLW87bekjhN7ybQ71i2fAv94k / pwRTTrvwb17eeTAAd4ZPZr + jcxOvvOHOwE455fnsHXrCiZNmsrw4VXcMOUOznvjaYanHWp58XXoTpuIiIiIiHSYkBBPOPvVrzzvwG3fDtdeC2lpMHcujB8Pjz4KhYXNb3vkopvpseMQZf16w + hRfPKLW3HV1ac97wd9 + vCDPn2YsXYtaUVFpzwu / op4sl7MwjnHoEHnERbmR // +W / i8 + kv02ZYFVVXNL9pLFNpERERERKRNxMfDl74Ejz0Gu3fD / ffDW2955ou76SbPu3HNecggKCyK5GdWkP3CU8Q9 / HdWjUvkYHrqac + 7
    uUcPHurfnznr1 / Npfn6Dx4SNDcNVOorTigGYNm0gFRXPkrpqAgcjjKyV7ze9UC9TaBMRERERkTbn6wsXXggvveSZG65 / f7jqKs90Ag8 / DKfIUg0aetF19NuWRfHYkfhPmMSHd3 / 5
    tHfdrurWjSeHDGHBxo28l5t70n4zI + 7
    yOLJezAJgzpyL2L37KVav8mH34EQy3nmpWd / Xm / ROm4iItAm909Y86iNFpCuqrvaMQPnoo5734BYtgsWLYdIksCb2INs + WEr5DddTFhJA / NOv0HvMjEaP / zAvjys2beKRQYNYFB9fb1 / +iny23LiFiZsmkpm5jf79BzFoUCXfGXMZwzJ2M / 7
    ttJZ + 1
    Vp6p01ERERERM4YPj6eQUqefx62bPEMVHL99TBqFPzpT9DADbGTDJy5iMFbsilIPo + Qqcl88N3Lqa6qPOXxM6KieGPUKL6xbRv / yMysty9iUgSV + ZUUpxeTmDiQPn2CSUrayPbwi0lI29Har9tiCm0iIiIiItLhEhLgzjs94e2Pf4Tly6FfP7jvPigvb / xcv4Agkv / 3
    vxQse4PYV97io8snNHr8uPBw3h89mp / s2sUf9 + +v3W4 + Rvzl8WS95HlEcvr0EVRWvsxnu75IVO5RKrO8M4Jkcym0iYiIiIhIp + HjA7NmwTPPeEadTE31jDqZevrxRug36UJ6fZxG3082svzB2xo9dkhoKB + dey5 / zsjgp7t3c + wR9fjL42vfa5s7dyG7dz / Npx / FsDkphN1vPdfq79cSCm0iIiIiItIp9e4Nr70Gd90Fl1ziuRNXUtL4OVE9 + lH2 / DMM + tn / sjWl8cFDkoKC + Ojcc3kxK4vHDx4EIHJqJOWZ5ZRsL2Hu3JvYuXMHCQmV7Bs4gNyUN7311ZpFoU1ERERERDotM7jmGs9dt / 37
    YfRo + OCDxs8ZNOsKtvz4FgKvvIr8zD2NHtstIICf9evHc1meu2vma8Qviif7pWwiIhIYNiyS3r03sCt + BsGpa731tZpFoU1ERERERDq9hATPI5MPPeSZsPvWW6Gg4NTHT / 3
    hw + yeMpwt8yc2OjAJwOyoKD4tKKCo0nPcsYm2AWbMGEdFxb9JLbySpK2HO2SSbYU2ERERERE5YyxYABs3erLTiBHw + uunPnbKc8sJLC7lw5vmNtpmuJ8fE8PDWZaXB0DkzEhKd5dSuqeUefO + xL59z / LhJ1M5HAq5a5Z78 + s0iUKbiIiIiIicUaKiPHO7 / f3vcNttcN11kJ198nH + QSF0f / 0j
    Br / 8
    AamP3ddom / NjY3njyBEAfPx8iF0YS9ZLWcyc + WUOH96Kr1WzrX8C + 95 + oQ2 + UeMU2kRERERE5Iw0ezZs2OB5dHLkSHjuOagZBLJWQv9RZD / 5
    MH2 / ex97Vi87ZVvzYmJ4Iyen / iiSL2UREBDEhAnd6NVrA7t6jaHi4w / b8is1SKFNRERERETOWKGh8NvfwtKl8NOfwmWXQc2YIrVGLrqZzbdcQenCiynJb + CWHDA0JAQHpNcMTxl9fjQl6SWUZZQxa9YUqqpeY71dROyG7W38jU6m0CYiIiIiIme8yZNhzRpISoKFC6GsrP7 + GQ8 + S9aAHqxdMAFXXX3S + WZW / xHJAB9iL40l6 + Us5s27nn37lvL2xquIyy6m6khOe3ylWgptIiIiIiJyVggMhN // HhIT4ZZb6j8qaT4 + jP33SmJ3HeLDO7 / Y4PnHHpE85tgjkuPHL6SychO5h2LY1CuYve + 82
    NZfpR6FNhEREREROWv4 + MBTT3nuuv3 + 9 / X3hUTGEfrqmwx79GU2vPTwSefOjopiZWEhhTVD / 0
    fPjaZoXRGVWZVMmdKLnj03sL3vORxZ9t / 2 + Cq1FNpEREREROSsEhYGr74Kv / oVvPVW / X29x8xgzx9 + SvzXbiNz65r65 / n5MTkionbof98gX2LnxZL9SjazZ8 + isvJ1NoVNJSC1 / nltTaFNRERERETOOklJ8PzzcP31sGVL / X3jv3Y3W66cxeGLkyk / WlRv37yYGF6v + 4
    hkzUTbl1yymIyM11iWeTl9Pj8IDbwX11YU2kRERERE5Kw0fTr84heeCblzc + vvm / HImxyNCGHFF6fU2z4vJoY3jhypHfo / 5
    qIYCj4roF / seCIj09iwaTq5QY6CDant9TUU2kRERERE5Ox1441w0UVw1VVQ86oaAD6 + fgz570qSVn7Ox / ffUrt9SEgIvmZsrhn63zfUl + gLosn + dzbTpvUjMWELm / vGs + fN59vtOyi0iYiIiIjIWe2hhzxPM37 / + / W3RyYmUfHCcwz55aNsfd8zIqSZnfyIZM0okhdcMI / q6jfZHDeK8o9T2q1 + hTYREZE2ZGa3mVm6maWZ2QMdXY + ISFfk5wfPPQevvQZPPFF / 38
    CZi9h040Ky7v1e7bZjj0geE3txLPkf5TN3 + tfJzHyT5UcvIGbD1vYqX6FNRESkrZhZMnApMNI5NxL4TcdWJCLSdcXEeEaUvOsu + OST + vuG / 88
    vGJG6h6IjmQDMjo4mtbCQgprnKf0i / IiaFYX / yhiSkjbw9qbLiT9URHV + XrvUrtAmIiLSdm4FHnDOVQI457I7uB4RkS5t6FD4 + 9 / hyith797j2 + P6DmXr4HjWP / ozAEJ9fTkvIoL36oxecuwRyZkzBxARdJTN3QPZ / 97
    SdqlboU1ERKTtDAJmmNmnZva + mY3v6IJERLq6 + fPhu9 + FhQuhuPj49vIvXUHAcy / Urp / 0
    iOSlseQty + P8GZdSVfUOm3r0I / u919qlZr92uYqIiMhZyszeAbrV3QQ44G48 / Wy0c26ymU0AngfOOVVbS5Ysqf2cnJxMcnJyG1QsIiJ33AFpaXDDDZ653Mxg9M33UPXTv5C9O524vkOZHxvLb / fvxzmHmeEf7U / k1Eh6Vn2BvLzvs9p3AmNXf3jaa6WkpJCSktKqeu3Y / APtwcxce15PREQ6jpnhnLOOrqMjmdnrwIPOuQ9q1rcDk5xzOQ0cqz5SRKQdlZbCrFkwbx7cc49n2ycz + lI5aSIzf / 0
    8
    zjn6r1zJv0eMYGRYGAAHHz / IkTePcO22 + RzNfYCP8r9CQl55s67bkv5Rj0eKiIi0nVeA2QBmNgjwbyiwiYhI + wsKgpdfhr / 9
    DV56ybMt4LobiFn6BnB86P96j0gujOXI20eYndyfrKND8C2voGzf7javVaFNRESk7TwJnGNmacC / gC93cD0iIlJH9 + 6
    wdCnccgts2ABjvvx9EjOL2bs2BYD5sbH1QltAXADhE8KZ1uNiqss / ZWOPMPYua / vBSBTaRERE2ohzrsI5d71zbqRzbvyxxyRFRKTzGDcO7r4bfvYz8A8KYXPycHY + fD8As6KiWFVn6H + A + Cvi6b9uFqWlKWyISCL / k / favEaFNhERERER6dJuuAHefRcOHYKYr3 + L3v / 5
    EFddTYivL1MiIni3ztD / cZfFUfB6MWPG7GR56Tj81m1o8 / oU2kREREREpEuLjIRFi + Af / 4
    ARC28ioLyaLe8 + B5z8iGRgYiBho8NIHhVPauEcem45CG08kJRCm4iIiIiIdHmLF8NjjwHmw46LJpL52G + BmvnacnKoO8Jv / OXxjCqexoG8EKis5OieHW1am0KbiIiIiIh0eZMmQXAwpKRAr1t / wOB31lBVUc7A4GACfXxIqzMTd9wX4kh8ZzwVpWtIS2z7wUgU2kREREREpMszg5tugkcfhQHTFpAfEciGF / 6
    MmZ30iGRQryCi + seR1DuN1SFJ5H + yrE1rU2gTEREREREBrrsO3ngDsrMh89JZFD35COB5RPL1nPrTbMZ9IY4JSTl8WnoeAevS2rQuhTYREREREREgOhoWLICnnoLB31rCiE + 2
    UVqUR3JUFGuKisivM / R / 9
    OxoRlcOYH3pBHptzWzTwUgU2kRERERERGosXux5RDJxyAR2J0Wy7slfEuLry7TIyHpD / 4
    eODmXIztnsPgRVroriXVvbrCaFNhERERERkRpTp4KPD3z8MRRdsQD39NPAyY9I + vj5cM6ocwkLXs / 6
    hHD2vPdSm9XUpNBmZheZ2edmttXMftDIcRPMrMLMvuC9EkVERERERNpH3QFJRt66hGHrMsg / tNcz9P + RI / WG / o + cHsnwvttZHdSPwuUpbVbTaUObmfkAfwYuBIYDV5vZkFMc9wDwlreLFBERERERaS / XXw + vvQbVQf1IH9WdDQ / fy8CQEEJ9fVlfVFR7XOSMSCZFGSvLziNg7cY2q6cpd9omAtucc3uccxXAs8DCBo67DXgROOzF + kRERERERNpVXBzMnw // 93 / grr6a0Bf / DVB7t + 2
    YiAkRDDswkXVHB9Nn + +E2G4ykKaGtJ7Cvzvr + mm21zKwHcJlz7i + Aea88ERERERGR9ndsQJIxN / 6
    EfrvyyNy65qTQ5hPow6ge89ifVUa5VVG4fXOb1OLnpXZ + D9R91 + 2
    UwW3JkiW1n5OTk0lOTvZSCSIi0pFSUlJISUnp6DJERES8YuZMKC + HdRujqJwygKo / 30
    fy717mS5s3k1dRQZS / PwBx07vRZ1c663wj6P3eS4wYONzrtZg7zS08M5sMLHHOXVSzfhfgnHMP1jlm57GPQBxQDCx2zr16QlvudNcTEZGzg5nhnNPTF02kPlJEpPP59a9h82a4bfavCL57CUP3lDB / wwa + mpjIlQkJABx5 + wg33X01o / OymTs5islPvddomy3pH5vyeGQqMMDMkswsALgKqBfGnHPn1Cz98LzX9o0TA5uIiIiIiMiZ5CtfgaVLoe + 8 / yE6r4ydK14 / +b228yIYV5LImqpxBKxtm8cjTxvanHNVwLeAt4FNwLPOuXQzu9nMFjd0ipdrFBERERERaXcJCTB3Ljz7fACfnz + GvX95oDa0Vdc8HeEX7seEiIWsKuhD3x1tMxhJk + Zpc8696Zwb7Jwb6Jx7oGbbI865Rxs49mvOuZe9XaiIiIiIiEh7OzZnW8JN36Hf6yvoHxRE + AlD // ebPIrCihKO + jnyt2zweg1NCm0iIiIiIiJd0fnnQ0EBFMVcQ7Wvsek / T5z0iGTkjEiGdN / B2pgI9rz3ktdrUGgTERERERE5BR8fuPFGeOxvPuy5eBo5j / 2
    R + bGxvJ6TU3tM5LRIJgYeJdV / MEWffOj9GrzeooiIiIiIyFnkq1 + FF1 + EhBt + zLCUjUwJCmBDcTG5FRUABMQFMCl4CmurhhG4Nt3r11doExERERERaUT37jBrFnz8 + fkcig9hy7O / Z3pkJO / k5tYeM3b0JawuSOCc3VleH4xEoU1EREREROQ0broJHnsMci67kNKnnuDCmJh6oS12Rjx + ASUU + UPOplVevbZCm4iIiIiIyGnMnQuHDwPn38fIlbsYRSWrCgtr90dOj2R09CHWREexb9lSr15boU1EREREROQ0fH3h61 + HZ / 8
    zgm0DY3H / +h1bSkooraoCIKh3ENNDo0kNGELBRx959doKbSIiIiIiIk3wta / Bc89B0ReuIPSZZxkUHMyG4uLa / VOHXM6air4Er // cq9dVaBMREREREWmCXr1g6lTYGncPg9MPM8KnmtV1HpHsO2ME6eVxDNyb49XBSBTaREREREREmmjxYnjy6R5sGtebhE / frvdeW9SMKKKDSskPhMPrV3jtmgptIiIiIiIiTTRvHuzbB9nTLmPku2 / WC23BA4OZHHGU1VGx7Fv2iteuqdAmIiIiIiLSRH5 + nnfbUg5 / iTkfrGLb0aMcrRmMxMyYM3ASq / z7k / fBx167pkKbiIhIGzGz0Wa2wszWmtlnZja + o2sSEZHW + / rX4e // Pg // qkr6WzXri4pq942bdglrK3sRnpbutesptImIiLSdXwH3OufOBe4Fft3B9YiIiBckJcGkST5s7tONfgd2srpOaIuZGcs + Yhh8IB9XcweutRTaRERE2k41EFnzOQrI6MBaRETEi669FtaGnsvAtNR677WFjQyjd2AVR4KMQ + s / 2
    pamKwAAD51JREFU8cq1FNpERETazneA35jZXjx33X7YwfWIiIiXzJwJ72bPY / JHy + sN + 2 + +xgX941gdEcfut70zGIlCm4iISCuY2TtmtqHOklbz81LgVuB251wfPAHuiY6tVkREvKV3b9heeRVzVqWzvaSEkjqPQl4w5Qus8u9L3vsfeeVafl5pRUREpItyzl1wqn1m9k / n3O01x71oZo831taSJUtqPycnJ5OcnOylKkVEpC1MmhbHvvf96VdRwvqiIs6L9DwR32f2EDY + kchl6e + TkpJCSkpKq65jzoszdZ / 2
    YmauPa8nIiIdx8xwzllH19GRzGwT8A3n3Admdj7wgHNuwimOVR8pInKG + etfIeyRUTx329eZO / dybuvVC4Dq8mqmDbuRN / c9SXhJJebrW3tOS / pHPR4pIiLSdm4CHjKztcDPgcUdXI + IiHjR9OnwScU0hq + tPxiJT4APoxMiyAr24cDqlFZfR6FNRESkjTjnljvnxjvnznXOneecW9vRNYmIiPcMHQrL877IrI9X1RuMBODSSWNZHZHAjtdfbfV1FNpERERERERawMcH + pw7gxG7drKjpITiOoORTJw / n9X + PTmybFnrr9PqFkRERERERLqoGTN82JwQS7 / iXNbVmWQ7eko0W3y6kbhjV6uvodAmIiIiIiLSQtOmQWrgKAbuSK / 3
    XptvqC + VwYkMzy7GVVa26hoKbSIiIiIiIi00bhyk5M9j7OqT32ubM7I / mSG + 7
    P3s3VZdQ6FNRERERESkhQICgHOuYu7KDaTm59fbt2DhQlaHx7Pp5ZdbdQ2FNhERERERkVaYND2RoMJM9hwtobDOo5C95wxiXWAihR + 836
    r2FdpERERERERaYfp02BCRRN / cw / UGI / GP8We3fyJJ + w + 0
    qn2FNhERERERkVaYPBk + LJvCkK0b6w1GAhCZOJgROSVUV5S3uH2FNhERERERkVaIiICDMVcwNXXtSYORfPHS2WSE + rH1gzdb3L5Cm4iIiIiISCsNnHoBE9K38OmR7Hrbp1x5AavDY0n951MtbluhTUREREREpJWmT / ehpKyIA2XlFNQZjCS4VzBpQYmwelWL21ZoExERERERaaXp02GlDadfVgZr6wxGApAV3pOBWYdb3LZCm4iIiIiISCt16wbpwRcyclPaSe + 1
    DTp3GiNzj1JVXtaithXaREREREREvCB8zDUkr9nIZ3m59bbfcNuX2Rvmz6dLn2tRuwptIiIiIiIiXjBldi8SM3fxadahetvjR / VgTXgMq // essFIFNpERERERES8YNo0yC7w5VA15NcZjMTM + DykG + E7t7SoXYU2ERERERERLxgwANb6TKJ / xm7WnPBeW2mPgQzPzT7FmY1TaBMREREREfECMygftIgJaRtPGozk / GuuZURuaYvaVWgTERERERHxkmGz5jMp / XM + ycyot / 2
    CryxgV3hAi9pUaBMREREREfGSmcl + +GcfYtUJI0j6 + vmyLiK6RW02KbSZ2UVm9rmZbTWzHzSw / xozW1 + zfGxmI1tUjYiIiIiIyBls1CjIOBJFDr7kVVTU27czqnuL2jxtaDMzH + DPwIXAcOBqMxtywmE7gRnOudHAz4HHWlSNiIiIiIjIGczPDw53n8Og3TtYU1RUb1 / C + Re3qM2m3GmbCGxzzu1xzlUAzwIL6x7gnPvUOZdfs / op0LNF1YiIiIiIiJzhuk27limbNp80yfbND / 28
    Re01JbT1BPbVWd9P46HsRuCNFlUjIiIiIiJyhps + tx999 + 7
    gg11bvdKen1daqWFms4CvAtO82a6IiIiIiMiZYuJEWJFZxPqj5V5prymhLQPoU2e9V822esxsFPAocJFzLvfE / ccsWbKk9nNycjLJyclNLFVERDqzlJQUUlJSOroMERGRDhcSAgUVSeT7 + nOkooIYf / 9
    WtWfOucYPMPMFtgDnAweBz4CrnXPpdY7pA7wHXO + c + 7
    SRttzpriciImcHM8M5Zx1dx5lCfaSIyNnl7q + +yNvjD / CLq69jTkxM7faW9I + nfafNOVcFfAt4G9gEPOucSzezm81scc1hPwFigIfNbK2ZfdacIkRERERERM4m4xcsYOLnW / hw / +5
    Wt3XaO23epL8iioh0HbrT1jzqI0VEzi45OfDbmxfy6Veu471Lr6zd3iZ32kRERERERKR5YmOhKtvYWN36sR8V2kRERERERNpAQMQoiv0DyKmoaFU7Cm0iIiKtZGZXmNlGM6sys7En7PuhmW0zs3Qzm9tRNYqISPs758JrGbN9G5 / lnnJw / SZRaBMREWm9NGAR8EHdjWY2FPgiMBSYh2fALr3nJyLSRcy + dDCDd27jnc2rW9WOQpuIiEgrOee2OOe2AScGsoV4Rl2udM7tBrYBE9u7PhER6Rh9 + kBIZg4rMw61qh2FNhERkbbTE9hXZz2jZpuIiHQRQaWRbAsJb1UbrR / KREREpAsws3eAbnU3AQ74sXPutY6pSkREOrveg2ZQGhBIVnk58QEBLWpDoU1ERKQJnHMXtOC0DKB3nfVeNdsatGTJktrPycnJJCcnt + CSIiLSmYQMiCf6Tw / xzbfeYFhMfIva0OTaIiLSJrri5Npm9j7wPefc6pr1YcDTwCQ8j0W + AwxsqDNUHykicnaqroZr77ydkAnjePyqL2tybRERkY5gZpeZ2T5gMvAfM3sDwDm3GXge2Ay8DnxDyUxEpGvx8YGwI0dJKyxpcRt6PFJERKSVnHOvAK + cYt8vgV + 2
    b0UiItKZJAb34t + xcS0 + X3faRERERERE2tD0OQsp9w / k4NGW3W1TaBMREREREWlDyZeMZvjObSxd / mGLztfjkSIiIiIiIm0oIAASDhwkxXxbdL7utImIiIiIiLSx2FIfdrUwfSm0iYiIiIiItLFR / Uaxt3vPFp2r0CYiIiIiItLGrr5qARX + / i06V6FNRERERESkjcUnhjNg984WnavQJiIiIiIi0g4Sso + 06
    DyFNhERERERkXbQNzCyReeZc87LpTRyMTPXntcTEZGOY2Y456yj6zhTqI8UETn7ZWXnkRAf3ez + UaFNRETahEJb86iPFBHpGlrSP + rxSBERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOrEmhTYzu8jMPjezrWb2g1Mc80cz22Zm68xsjHfLFBER6bzM7Aoz22hmVWY2ts72OWa2yszWm1mqmc3qyDpFROTMdNrQZmY + wJ + BC4HhwNVmNuSEY + YB / Z1zA4Gbgb + 2
    Qa1dTkpKSkeXcEbR76t59PtqPv3OpBFpwCLggxO2ZwGXOOdGAzcA / 2
    znus5a + vexefT7ah79vppHv6 + 215
    Q7bROBbc65Pc65CuBZYOEJxywEngJwzq0EIs2sm1cr7YL0L0Dz6PfVPPp9NZ9 + Z3IqzrktzrltgJ2wfb1zLrPm8yYgyMz8O6LGs43 + fWwe / b6aR7 + v5tHvq + 01J
    bT1BPbVWd9fs62xYzIaOEZERKTLMrMrgDU1fwAVERFpMr + OLkBERORMYGbvAHWfIjHAAT92zr12mnOHA78ELmi7CkVE5GxlzrnGDzCbDCxxzl1Us34X4JxzD9Y55q / A + 86552
    rWPwdmOucOndBW4xcTEZGzinPOTn / U2cPM3gfucM6tqbOtF / Ae8BXn3KeNnKs + UkSki2hu / 9
    iUO22pwAAzSwIOAlcBV59wzKvAN4HnakJe3omBrSXFiYiInIFq + zoziwT + A / ygscAG6iNFROTUTvtOm3OuCvgW8DawCXjWOZduZjeb2eKaY14HdpnZduAR4BttWLOIiEinYmaXmdk + YDLwHzN7o2bXt4D + wD1mttbM1phZXIcVKiIiZ6TTPh4pIiIiIiIiHadJk2t7Q1Mm6BYPM + tlZsvMbJOZpZnZtzu6pjOBmfnU / BX71Y6upbMzs0gze8HM0mv + OZvU0TV1Zmb2nZqJkzeY2dNmFtDRNXUmZva4mR0ysw11tkWb2dtmtsXM3qp5TFAaoP6x6dQ / toz6x + ZRH9k86iMb560 + sl1CW1Mm6JZ6KoHvOueGA + cB39Tvq0luBzZ3dBFniD8ArzvnhgKjgfQOrqfTMrMewG3AWOfcKDzvAl / VsVV1Ok / i + f / 3
    uu4C3nXODQaWAT9s96rOAOofm039Y8uof2we9ZFNpD6ySbzSR7bXnbamTNAtNZxzmc65dTWfi / D8n4XmvWtEzehs84G / dXQtnZ2ZRQDTnXNPAjjnKp1zBR1cVmfnC4SamR8QAhzo4Ho6Fefcx0DuCZsXAv + o + fwP4LJ2LerMof6xGdQ / Np / 6
    x + ZRH9ki6iMb4a0 + sr1CW1Mm6JYGmFlfYAywsmMr6fR + B9yJZ84kaVw / INvMnqx5XOZRMwvu6KI6K + fcAeAhYC + QgWd03Hc7tqozQsKxUYSdc5lAQgfX01mpf2wh9Y9Npv6xedRHNoP6yBZrdh / Zbu + 0
    SfOZWRjwInB7zV8UpQFmdjFwqOavr0ad4balQX7AWOB / nXNjgRI8t + mlAWYWhecvYklADyDMzK7p2KrOSPoPRvEa9Y9No / 6
    xRdRHNoP6SK85bR / ZXqEtA + hTZ71XzTY5hZpbzC8C / 3
    TO / buj6 + nkpgILzGwn8Awwy8ye6uCaOrP9wD7n3Kqa9RfxdFDSsDnATufckZopUF4GpnRwTWeCQ2bWDcDMEoHDHVxPZ6X + sZnUPzaL + sfmUx / ZPOojW6bZfWR7hbbaCbprRpS5Cs + E3HJqTwCbnXN / 6
    OhCOjvn3I + cc32cc + fg + WdrmXPuyx1dV2dVczt + n5kNqtl0PnpBvTF7gclmFmRmhuf3pZfST3biX / FfBW6o + fwVQP9x3TD1j82n / rGJ1D82n / rIZlMf2TSt7iP9vF / TyZxzVWZ2bIJuH + Bx55z + Bz0FM5sKXAukmdlaPLdMf + Sce7NjK5OzyLeBp83MH9gJfLWD6 + m0nHOfmdmLwFqgoubnox1bVediZv8CkoFYM9sL3As8ALxgZl8D9gBf7LgKOy / 1j
    82j / lHaifrIJlIfeXre6iM1ubaIiIiIiEgnpoFIREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxBTaREREREREOjGFNhERERERkU5MoU1ERERERKQTU2gTERERERHpxP4f8C2WWL + gKy0AAAAASUVORK5CYII =
    def create_all_even_vectors(n):
        ret = []
        for i in range(2 ** n):
            binformat = bin(i)[2:].zfill(n)
            if 0 == binformat.count('0') % 2:
                ret.append(create_vector_from_string(binformat))
        return ret


    evens = create_all_even_vectors(n)
    IDeven = sum(v * v.trans() for v in evens)
    PRECISION = 2 ** -40

    alist = []
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = random.uniform(-10, 10)
            alist.append(a)
            terms.append(XXZZham.XXZZ_term(i, j, a))

    H = XXZZham.XXZZham(terms)
    H_com = H.get_commuting_term_ham()
    H_highE = add_high_energies(rotate_to_00_base(H_com), 30)
    h_com_rot = rotate_to_00_base(H_com)

    ID2n = tensor([qeye(2)] * n * 2)
    HE_ev = H_highE.eigenstates(eigvals=1)[1][0]
    P_HE = ID2n - HE_ev * HE_ev.trans()data:image/png;base64,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
    HCR_en, HCR_ev = h_com_rot.eigenstates(eigvals=10)
    HCR_degeneracy = sum(abs(HCR_en - HCR_en.min()) < PRECISION)
    HCR_groundspace = HCR_ev[0:HCR_degeneracy]
    # _, HCR_groundspace = h_com_rot.eigenstates(eigvals=HCR_degeneracy)
    P_CR = ID2n - sum([gs * gs.trans() for gs in HCR_groundspace])

    # prepare the first state for CLH-HIGHE evolution
    tlist = np.linspace(0, 10, 10)
    P99 = P_HE * 0.01 + P_CR * 0.99
    psi0_99 = P99.eigenstates(eigvals=1)[1][0]

    P_mat, _, psis = asim.sim_degenerate_adiabatic(tlist, P99, P_CR, psi0_99, 10)
    print(P_mat[-1][0], "this should be close to one")
    psifinal_from_99 = psis[-1]  # should be the GS we get from evolution to P_CR starting in 99
    psi0_small = H.get_ham().eigenstates(eigvals=1)[1][0]

    uniform_over_left_0000 = sum(create_all_even_vectors(n))
    uniform_over_left_0000/=uniform_over_left_0000.norm()


    left_extended_to_even_psi0 = tensor([uniform_over_left_0000, psi0_small])

    print(abs(left_extended_to_even_psi0.overlap(psifinal_from_99)))
# -*- coding: utf-8 -*-

# Copyright 2019 Jean-Baptiste Delisle
# Modified 2022 Flavien Kiefer

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class rvModel():
  def __init__(self, t, rv, *args, **kwargs):
    self.rv = rv
    self.t = t

  def wav_to_dvel(wav,c):
      dvel = (wav[1:] - wav[:-1]) / (wav[1:]) * c
      return dvel

  def loglambda(wav0, flux0,c):
    assert wav0.shape == flux0.shape
    npix = wav0.size
    wav = np.logspace(np.log10(wav0[0]), np.log10(wav0[-1]), wav0.size)
    spline = InterpolatedUnivariateSpline(wav0, flux0)
    flux = spline(wav)
    dvel = rvModel.wav_to_dvel(wav,c)
    dvel = np.mean(dvel)
    return wav, flux, dvel

  def CCF(flux, ref_flux, nwav, dvel, ref_wav, wav):
    ref_spline = InterpolatedUnivariateSpline(ref_wav, ref_flux)
    ref_flux = ref_spline(wav)
    flux -= np.mean(flux)
    ref_flux -= np.mean(ref_flux)
    lag = np.arange(-nwav + 1, nwav) 
    dvel = -1.0 * lag * dvel
    a = ref_flux
    b = flux
    a=(a-np.min(a))/(np.max(a)-np.min(a))
    b=(b-np.min(b))/(np.max(b)-np.min(b))
    f_a=np.fft.fft(a)
    f_b=np.fft.fft(b)
    f_a_c=np.conj(f_a)
    c_corr=np.fft.ifft(f_a_c*f_b)
    c_corr=np.abs(np.roll(c_corr,len(c_corr) // 2))
    corr=(c_corr-np.min(c_corr))/(np.max(c_corr)-np.min(c_corr))
    s = int(len(corr)/2)
    e = -s+1
    dv = -dvel[s:e]
    return dv,corr

  
  def dlamb(corr, dv, c):
    for i in range(len(corr)-1):
        if corr[i]==max(corr) :
            imax=i
    dlambda = dv[imax]*5000/c
    return dlambda,imax

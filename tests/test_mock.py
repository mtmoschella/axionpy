import unittest
import axionpy.units as u
import axionpy.axis as ax
from axionpy import mock
import numpy as np
import matplotlib.pyplot as plt
import os

class TestMock(unittest.TestCase):
    def test_short_signals(self):
        m = 1.*u.Hz
        g = 1.0e-8*u.GeV**-1

        coherence_time = 1.e6/m
        
        fsample = 10.*m
        tmax = 1.e-3*coherence_time
        
        dt = 1./fsample
        t = np.arange(0., tmax.to_value(u.s), dt.to_value(u.s))*u.s
        N = len(t)
        
        lat, lon = 40., 75.
        theta, phi = 0.0, 0.0
        
        S1 = mock.signal(m, g, t=t, lat=lat, lon=lon, theta=theta, phi=phi)
        S2 = mock.signal(m, g, t=t, lat=lat, lon=lon, theta=theta, phi=phi, ntrials=10)
        
        self.assertTrue(S1.unit==u.GeV)
        self.assertTrue(S2.unit==u.GeV)
        self.assertTrue(np.shape(S1)==(N,))
        self.assertTrue(np.shape(S2)==(10,N))

    def test_long_signals(self):
        m = 1.*u.Hz
        g = 1.0e-8*u.GeV**-1
        
        coherence_time = 1.e6/m
        
        fsample = 10.*m
        tmax = 0.1*coherence_time
        
        dt = 1./fsample
        N = int((tmax/dt).to_value(u.dimensionless_unscaled))
        
        lat, lon = 40., 75.
        theta, phi = 0.0, 0.0
        axis = ax.Axis(lat, lon, theta, phi)
        components = axis.components(N=N, dt=dt, basis='xyz', verbose=True)
        
        testdir = 'data/'
        os.system('mkdir -p '+testdir)
        f1 = testdir+'signal_test_1.memmap'
        f2 = testdir+'signal_test_2.memmap'
        
        S1 = mock.signal(m, g, N=N, dt=dt, axis=axis, verbose=True, fname=f1)
        S2 = mock.signal(m, g, N=N, dt=dt, components=components, verbose=True, ntrials=10, fname=f2)
        self.assertTrue(np.shape(S1)==(N,))
        self.assertTrue(np.shape(S2)==(10,N))
        
        os.system('rm -rfv '+testdir)
        
if __name__=='__main__':
    unittest.main()

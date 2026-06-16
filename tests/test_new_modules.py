import numpy as np
from dac.modules.timedata import TimeData
from dac.modules.nvh.data import FreqDomainData, FreqIntermediateData, DataBins, OrderList, OrderInfo, OrderSliceData
from dac.modules.drivetrain import GearStage, BallBearing, GearboxDefinition, BearingInputStage


class TestTimeDataNewMethods:
    def test_integrate_returns_timedata(self):
        td = TimeData('test', y=np.ones(100), dt=0.01, y_unit='m/s2')
        result = td.integrate()
        assert result.name == 'test-IntT'
        assert result.y_unit == 'm/s2*s'
        assert len(result.y) == 100

    def test_differentiate_returns_timedata(self):
        td = TimeData('test', y=np.linspace(0, 10, 200), dt=0.01, y_unit='m/s2')
        result = td.differentiate()
        assert result.name == 'test-DiffT'
        assert result.y_unit == 'm/s2/s'
        assert len(result.y) == 200

    def test_statistics_keys(self):
        td = TimeData('test', y=np.random.randn(500), dt=0.01, y_unit='Pa')
        stats = td.statistics()
        expected_keys = {'name', 'mean', 'std', 'min', 'max', 'rms', 'crest_factor', 'skewness', 'kurtosis'}
        assert set(stats.keys()) == expected_keys

    def test_statistics_rms(self):
        td = TimeData('test', y=np.array([3.0, 4.0]), dt=0.01)
        stats = td.statistics()
        expected_rms = np.sqrt(np.mean(np.array([3.0, 4.0])**2))
        assert abs(stats['rms'] - expected_rms) < 1e-10

    def test_dcoffset_removal(self):
        td = TimeData('test', y=np.array([5.0, 7.0, 9.0]), dt=0.01)
        y_dc = td.y - np.mean(td.y)
        assert abs(np.mean(y_dc)) < 1e-10


class TestFreqDomainData:
    def test_effective_value_with_fmin_fmax(self):
        y = np.ones(100, dtype=complex)
        fd = FreqDomainData('test', y=y, df=0.5, y_unit='Pa')
        full = fd.effective_value()
        part = fd.effective_value(fmin=10, fmax=20)
        assert 0 < part < full
        assert fd.effective_value(fmin=0, fmax=0) == full  # (0,0) = full range

    def test_as_timedomain(self):
        y = np.ones(50, dtype=complex) * (1 + 1j)
        fd = FreqDomainData('test', y=y, df=1.0, y_unit='m/s2')
        td = fd.as_timedomain()
        assert isinstance(td, TimeData)
        assert td.name == 'test-AsT'
        assert td.y_unit == 'm/s2'

    def test_to_timedomain_existing(self):
        y = np.ones(50, dtype=complex) * (1 + 1j)
        fd = FreqDomainData('test', y=y, df=1.0, y_unit='m/s2')
        td = fd.to_timedomain()
        assert isinstance(td, TimeData)
        assert len(td.y) > 0


class TestFreqIntermediateData:
    def test_rectify_to(self):
        z = np.ones((20, 100), dtype=complex)
        ref = DataBins('ref', y=np.arange(20) * 60, y_unit='rpm')
        fid = FreqIntermediateData('test', z=z, df=0.5, z_unit='m/s2', ref_bins=ref)
        result = fid.rectify_to(5, 2)
        assert result.z.shape == (10, 20)
        assert result.df == 2.5

    def test_rectify_to_noop(self):
        z = np.ones((2, 4), dtype=complex)
        ref = DataBins('ref', y=np.array([10, 20]), y_unit='rpm')
        fid = FreqIntermediateData('test', z=z, df=0.5, ref_bins=ref)
        result = fid.rectify_to(50, 50)
        assert result is fid


class TestOrderSliceData:
    def test_rectify2freqdata(self):
        from dac.modules.nvh.data import SliceData
        src = FreqIntermediateData('src', z=np.ones((5, 20), dtype=complex), df=0.5, z_unit='m/s2')
        osd = OrderSliceData('osd', source=src)
        oi = OrderInfo('test_order', 1.0)
        osd.slices[oi] = SliceData(
            f=np.linspace(0, 50, 30),
            ref=np.linspace(100, 600, 30),
            amplitude=np.random.randn(30),
        )
        results = osd.rectify2freqdata(df=1.0)
        assert len(results) == 1
        assert isinstance(results[0], FreqDomainData)
        assert results[0].name == 'test_order'


class TestGearStage:
    def test_parallel_stage(self):
        gs = GearStage({'Wheel': 60, 'Pinion': 20})
        assert gs.ratio == 3.0
        assert gs.stage_type == GearStage.StageType.Parallel

    def test_planetary_stage(self):
        gs = GearStage({'RG': 60, 'PG': 20, 'SU': 20, 'NoP': 3})
        assert gs.ratio == 4.0
        assert gs.stage_type == GearStage.StageType.Planetary

    def test_get_freq_at_order_parallel(self):
        gs = GearStage({'Wheel': 50, 'Pinion': 25})
        speed = 3000
        assert gs.get_freq_at_order('f', speed) == 50.0
        assert gs.get_freq_at_order('fz', speed) == 2500.0
        assert gs.get_freq_at_order('nonexistent', speed) is None

    def test_get_order_name_at_freq(self):
        gs = GearStage({'Wheel': 50, 'Pinion': 25})
        speed = 3000
        fz = gs.fz(speed)
        name = gs.get_order_name_at_freq(fz, speed)
        assert name == '1fz'

    def test_get_order_name_at_freq_no_match(self):
        gs = GearStage({'Wheel': 50, 'Pinion': 25})
        speed = 3000
        name = gs.get_order_name_at_freq(99999.0, speed)
        assert name is None


class TestGearboxDefinition:
    def test_total_ratio(self):
        gb = GearboxDefinition('test', stages=[
            GearStage({'Wheel': 60, 'Pinion': 20}),
            GearStage({'Wheel': 40, 'Pinion': 10}),
        ])
        assert gb.total_ratio == 12.0

    def test_get_freqs_labels_at(self):
        gb = GearboxDefinition('test', stages=[
            GearStage({'Wheel': 50, 'Pinion': 25}),
            GearStage({'Wheel': 40, 'Pinion': 20}),
        ])
        freqs = gb.get_freqs_labels_at(3000)
        assert len(freqs) > 0
        labels = [lbl for _, lbl in freqs]
        assert 'f_1' in labels
        assert 'fz_1' in labels


class TestBallBearing:
    def test_get_freqs_labels_at(self):
        bb = BallBearing('SKF6205', N_balls=9, D_ball=7.94, D_pitch=39.04, beta=0)
        freqs = bb.get_freqs_labels_at(1800)
        assert len(freqs) == 4
        labels = [lbl for _, lbl in freqs]
        assert 'bpfo' in labels
        assert 'bpfi' in labels
        assert 'bsf' in labels
        assert 'ftf' in labels
        for freq, _ in freqs:
            assert freq >= 0

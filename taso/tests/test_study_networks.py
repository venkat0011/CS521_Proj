"""Check explicit initialization without a GPU or model calls."""
import hashlib
import math
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'secondary_experiment'))
sys.path.insert(0,str(ROOT/'benchmarks'))
from test_benchmark_suite import ShapeGraph,Tensor
from study_networks import build_study_network,_secondary_weight
try:
    import numpy as np
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy required; separately verified with project Python')
class NetworkInitializationTests(unittest.TestCase):
    def test_all_three_architectures_and_reproducible_weights(self):
        class CapturingGraph(ShapeGraph):
            def __init__(self):super().__init__();self.hashes=[]
            def new_weight(self,*,dims,data=None):
                self.parameters+=math.prod(dims)
                if data is not None:
                    assert data.dtype==np.float32 and tuple(data.shape)==tuple(dims)
                    assert np.isfinite(data).all()
                    bound=math.sqrt(6/math.prod(dims[1:]))
                    assert float(abs(data).max())<=bound+1e-6
                    self.hashes.append(hashlib.sha256(data.tobytes()).hexdigest())
                return Tensor(dims)
        ts=types.SimpleNamespace(new_graph=CapturingGraph)
        for name in ('squeezenet','resnext50','inceptionv3'):
            with self.subTest(network=name):
                original,_,_=build_study_network(name,ts,ROOT/'TASO')
                a,source,info=build_study_network(name,ts,ROOT/'TASO',initialization='fan_in')
                b,_,_=build_study_network(name,ts,ROOT/'TASO',initialization='fan_in')
                self.assertEqual((original.ops,original.parameters,original.last.dims),(a.ops,a.parameters,a.last.dims))
                self.assertTrue(a.hashes);self.assertEqual(a.hashes,b.hashes)
                self.assertEqual(info['weight_initialization'],'fan_in_uniform_relu')
                self.assertIn('_secondary_weight',source)
    def test_explicit_data_is_preserved(self):
        data=np.array([1,-2],dtype=np.float32)
        class G:
            def new_weight(self,**kw):return kw['data']
        self.assertIs(_secondary_weight(G(),dims=(2,),data=data),data)

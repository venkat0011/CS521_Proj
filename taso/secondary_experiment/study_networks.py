"""Explicit secondary-study weight initialization; historical builders stay unchanged."""
import ast
import hashlib
import inspect
from networks import build_network, construction_source


def _secondary_weight(graph, *, dims, data=None):
    import math
    import numpy as np
    if data is None:
        fan_in=max(1,math.prod(dims[1:]))
        bound=math.sqrt(6.0/fan_in)
        data=np.random.uniform(-bound,bound,size=dims).astype(np.float32)
    return graph.new_weight(dims=dims,data=data)


def build_study_network(name,ts,repo,seed=521,initialization='original'):
    if initialization=='original':
        graph,source,info=build_network(name,ts,repo,seed)
        info['weight_initialization']='original_uniform_0_1'
        return graph,source,info
    if initialization!='fan_in':raise ValueError('Unknown secondary weight initialization')
    import numpy as np
    np.random.seed(seed)
    source,info=construction_source(name,repo)
    class Rewriter(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='new_weight':
                return ast.copy_location(ast.Call(func=ast.Name(id='_secondary_weight',ctx=ast.Load()),
                    args=[node.func.value]+node.args,keywords=node.keywords),node)
            return node
    tree=ast.fix_missing_locations(Rewriter().visit(ast.parse(source)))
    executed=inspect.getsource(_secondary_weight)+'\n\n'+ast.unparse(tree)+'\n'
    namespace={'ts':ts,'taso':ts}
    exec(compile(executed,f'<secondary:{name}>','exec'),namespace)
    builder=namespace['squeezenet'] if name=='squeezenet' else namespace['build']
    graph=builder(ts,batch_size=1)
    info.update(network=name,weight_seed=seed,pretrained=False,
        weight_initialization='fan_in_uniform_relu',
        weight_initialization_scope='Explicit new secondary-study workload; same seeded weights for TASO and Kimi; historical weights unchanged',
        historical_construction_sha256=hashlib.sha256(source.encode()).hexdigest(),
        construction_sha256=hashlib.sha256(executed.encode()).hexdigest())
    return graph,executed,info

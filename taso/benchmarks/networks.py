"""Native TASO benchmark graphs; importing this module never starts a search."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

NETWORKS = ('squeezenet', 'resnext50', 'inceptionv3')


def squeezenet(ts, batch_size=1):
    """SqueezeNet 1.1 layout in TASO, without biases, dropout or final flatten.

    Fire widths and pooling positions follow torchvision's SqueezeNet 1.1.
    At input 224, VALID pooling has the same shapes as its ceil-mode pools.
    Output stays [N,1000,1,1]. No pretrained weights are loaded.
    """
    graph = ts.new_graph()
    x = graph.new_input(dims=(batch_size, 3, 224, 224))

    def conv(x, channels, kernel, stride=1, padding='SAME'):
        w = graph.new_weight(dims=(channels, x.dim(1), kernel, kernel))
        return graph.relu(graph.conv2d(input=x, weight=w,
                          strides=(stride, stride), padding=padding))

    def pool(x):
        return graph.maxpool2d(input=x, kernels=(3, 3), strides=(2, 2), padding='VALID')

    x = pool(conv(x, 64, 3, 2, 'VALID'))
    widths = [(16, 64), (16, 64), (32, 128), (32, 128),
              (48, 192), (48, 192), (64, 256), (64, 256)]
    for index, (squeeze, expand) in enumerate(widths):
        squeezed = conv(x, squeeze, 1)
        x = graph.concat(1, [conv(squeezed, expand, 1), conv(squeezed, expand, 3)])
        if index in (1, 3):
            x = pool(x)
    x = conv(x, 1000, 1)
    graph.avgpool2d(input=x, kernels=(x.dim(2), x.dim(3)),
                    strides=(1, 1), padding='VALID')
    return graph


def construction_source(name, repo):
    """Extract construction only; upstream examples have executable top levels.

    Compile the extracted source itself, which is also saved in the LLM prompt.
    No optimizer/export/print statements from those examples are executed.
    """
    if name not in NETWORKS:
        raise ValueError(f'Unknown network: {name}')
    if name == 'squeezenet':
        return inspect.getsource(squeezenet), {
            'variant': 'SqueezeNet 1.1 layout, bias-free TASO variant',
            'input_shape': [1, 3, 224, 224],
            'notes': ['No biases or pretrained weights; inference dropout omitted.',
                      'Output remains NCHW; no final flatten.',
                      'Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py'],
        }
    path = Path(repo) / 'examples' / f'{name}.py'
    text = path.read_text()
    tree = ast.parse(text)
    definitions = [ast.get_source_segment(text, node) for node in tree.body
                   if isinstance(node, ast.FunctionDef)]
    if name == 'inceptionv3':
        wrapper = 'def build(ts, batch_size=1):\n    return inception_v3(batch_size=batch_size)\n'
        notes = ['Bundled TASO Inception-v3 graph at batch 1 (upstream calls batch 32).',
                 'Preserves upstream 1x1 final pooling and absence of classification head.']
        shape = [1, 3, 299, 299]
    else:
        start = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'graph' for t in node.targets))
        stop = next(i for i, node in enumerate(tree.body[start:], start)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'optimize')
        body = '\n'.join(ast.get_source_segment(text, node) for node in tree.body[start:stop])
        # Both checked-in examples use this exact input. Fail visibly if they change.
        if body.count('(1,3,224,224)') != 1:
            raise ValueError(f'Unexpected input declaration in {path}')
        body = body.replace('(1,3,224,224)', '(batch_size,3,224,224)')
        wrapper = 'def build(ts, batch_size=1):\n' + textwrap.indent(body, '    ') + '\n    return graph\n'
        shape = [1, 3, 224, 224]
        notes = (['Bundled TASO ResNeXt-50: 3/4/6/3 blocks, groups 32.',
                  'Preserves upstream projection ReLU; no classification head.']
                 if name == 'resnext50' else
                 ['Bundled TASO NASNet-A: 15 normal and 3 reduction cells; no classification head.',
                  'This is not identified as NASNet-A Large (89M parameters).'])
    return '\n\n'.join(definitions + [wrapper]), {
        'variant': f'TASO bundled {name}', 'upstream_file': f'examples/{name}.py',
        'upstream_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'input_shape': shape, 'notes': notes,
    }


def build_network(name, ts, repo, seed=521):
    import numpy as np
    np.random.seed(seed)
    source, info = construction_source(name, repo)
    namespace = {'ts': ts, 'taso': ts}
    exec(compile(source, f'<benchmark:{name}>', 'exec'), namespace)
    builder = namespace['squeezenet'] if name == 'squeezenet' else namespace['build']
    graph = builder(ts, batch_size=1)
    info.update(network=name, weight_seed=seed, pretrained=False,
                construction_sha256=hashlib.sha256(source.encode()).hexdigest())
    return graph, source, info

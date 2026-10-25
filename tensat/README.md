# Tensat environment scaffold

Create a separate Python environment from the project root:

```sh
python3 setup_env.py tensat
source tensat/.venv/bin/activate
python --version
```

The interpreter is pinned in `.python-version`. This creates an empty Python
scaffold, not a Tensat installation. The Tensat source, native toolchain and
package requirements will be configured after inspecting that repository.
Its Python version can be adjusted independently of TASO's.

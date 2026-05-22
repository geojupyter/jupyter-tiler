# Jupyter Tiler

[![PyPI - Version](https://img.shields.io/pypi/v/jupyter-tiler)](https://pypi.org/project/jupyter-tiler/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/jupyter-tiler)](https://pypi.org/project/jupyter-tiler/)
![License](https://img.shields.io/github/license/geojupyter/jupyter-tiler)
[![ReadTheDocs Status](https://readthedocs.org/projects/jupyter-tiler/badge/?version=latest)](https://jupyter-tiler.readthedocs.io)
[![Github Actions Status - Build](https://github.com/geojupyter/jupyter-tiler/workflows/Build/badge.svg)](https://github.com/geojupyter/jupyter-tiler/actions/workflows/build.yml)
[![Github Actions Status - Typecheck](https://github.com/geojupyter/jupyter-tiler/workflows/Typecheck/badge.svg)](https://github.com/geojupyter/jupyter-tiler/actions/workflows/typecheck.yml)

> [!IMPORTANT]
> This repository is experimental and in the prototype stage.
> Expect bugs.
> Expect a possible pivot and/or name change in the future :smile:
>
> Your feedback and contributions are welcome!
> Please open an issue, DM Matt Fisher, or post in the `#geojupyter` channel on the [Jupyter Zulip](https://jupyter.zulipchat.com)!

A Jupyter server extension which provides an API to launch a server to dynamically tile
[Xarray DataArray](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html)s
for interactive visualization.

## Who is this for?

Intended to be consumed by interactive map libraries for Jupyter, **not end-users**,
e.g.:

* [Leafmap](https://leafmap.org/)
* [ipyleaflet](https://ipyleaflet.readthedocs.io)
* [ipyopenlayers](https://ipyopenlayers.readthedocs.io/)
* More?

## What problem does this solve?

For authors of interactive map libraries for Jupyter, providing a dynamic HTTP tile
server presents a unique problem: **they don't know where Jupyter is running**.
It could be, for example, running on:

* users' local machines
* a shared JupyterLab instance on an intranet
* an authenticated JupyterHub in a public cloud

The first case is the simplest; when the tile server is running on `localhost`, the map
viewer running in JavaScript in the user's browser can connect to it.

In the other cases, the map viewer needs a public URL to connect to.
The URL of the current JupyterHub instance may not be known.
Additionally, a map server running in a Jupyter kernel isn't exposed to the public
internet in many cases (for example, when it's running in a Kubernetes pod as part of a
JupyterHub).
This extension provides [dynamic proxying](https://jupyter-server-proxy.readthedocs.io/)
to map servers running in the kernel.

## Usage

As a Jupyter interactive map library author, you may implement a method like:

```python
from jupyter_tiler.titiler import add_data_array


class MyMapLibrary:
  # ...

  def add_xarray_layer(self, da: xr.DataArray):
    # Add the layer to the tile server.
    # The server will be started the first time this is called.
    # A URL that passes through the Jupyter server proxy will be returned:
    url = add_data_array(da)

    # Add the layer to your map!
    self._add_tile_layer(url)
```

## Install

Recommended:

```bash
uv add jupyter-tiler
```

Or:

> [!WARNING]
> Installation with pixi/conda/mamba/micromamba is not yet supported.
> Please use another installation method!

```bash
pixi add jupyter-tiler
```

For other methods of installation, including pip, conda, mamba, and micromamba, see the
[installation instructions in the documentation](https://jupyter-tiler.readthedocs.io/en/latest/user-guide/install/).

### From source

```bash
uv add git+https://github.com/geojupyter/jupyter-tiler.git#egg=jupyter-tiler
```

For _development_ instructions, please view the
[development install instructions in our documentation's Contributor Guide](https://jupyter-tiler.readthedocs.io/en/latest/contributor-guide/how-tos/development-install/)!

## Contributing

Please see the
[Contributor Guide in our documentation](https://jupyter-tiler.readthedocs.io/en/latest/contributor-guide/)!

## :rocket: Powered by...

[TiTiler (Development Seed)](https://developmentseed.org/titiler/)

Other backends (e.g.
[xpublish-tiles (earthmover)](https://github.com/earth-mover/xpublish-tiles)) may be
supported in the future!

## :sparkles: Inspired by...

[jupytergis-tiler](https://github.com/geojupyter/jupytergis-tiler) by
[David Brochart](https://github.com/davidbrochart)

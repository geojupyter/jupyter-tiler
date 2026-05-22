# `jupyter-tiler` Documentation

```{toctree}
:maxdepth: 1
:hidden:

user-guide/index
contributor-guide/index
```

Thank you for trying out `jupyter-tiler`!

![High-level diagram of `jupyter-tiler`](/assets/images/high-level-diagram.svg)

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

## Getting started

Check out our [installation instructions](/user-guide/install)
and [quickstart instructions](/user-guide/quickstart)!

:::{warning}
This repository is experimental and in the prototype stage.
Expect bugs.
Expect a possible pivot and/or name change in the future 😄

Your feedback and contributions are welcome!
Please open an issue, DM Matt Fisher, or post in the `#geojupyter` channel on the [Jupyter Zulip](https://jupyter.zulipchat.com)!
:::

## 🚀 Powered by...

[TiTiler (Development Seed)](https://developmentseed.org/titiler/)

Other backends (e.g.
[xpublish-tiles (earthmover)](https://github.com/earth-mover/xpublish-tiles)) may be
supported in the future!

## ✨ Inspired by...

[jupytergis-tiler](https://github.com/geojupyter/jupytergis-tiler) by
[David Brochart](https://github.com/davidbrochart)

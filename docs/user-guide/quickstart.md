# Quickstart

As an author of a an interactive map library for Jupyter, you might use
`jupyter-tiler` to provide the ability to dynamically visualize data in Xarray
DataArrays without writing to a file like so:

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

import cairo
import numpy as np
from gi import require_version
import gi

from utils import reshape

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg


def rasterize_in_memory(xml_desc):
    img = cairo.ImageSurface(cairo.FORMAT_A8, 28, 28)
    ctx = cairo.Context(img)
    handle = Rsvg.Handle.new_from_data(xml_desc.encode())
    handle.render_cairo(ctx)
    buf = img.get_data()
    img_array = np.ndarray(shape=(28, 28),
                           dtype=np.uint8,
                           buffer=buf)

    img_array = reshape(img_array)

    return img_array

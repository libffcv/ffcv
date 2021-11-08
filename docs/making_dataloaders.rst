Making dataloaders
------------------

.. code-block:: python

    # FFCV-specific imports
    from ffcv.loader import Loader, OrderOption
    from ffcv import transforms
    from ffcv.fields.rgb_image import SimpleRGBImageDecoder
    from ffcv.fields.basics import IntDecoder
    from torchvision.transforms import Normalize

Simple train loader


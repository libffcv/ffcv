Working with Image Data in FFCV
===============================

Images can often be responsible for the majority of resources (storage and/or
compute) consumed by computer vision datasets.
FFCV offers a wide range of options to control the storage and retrieval of
images, allowing the user to cater to the specific needs of each project and
hardware configuration.

.. note::

  This page is specifically about the options and API for writing and reading
  image data with FFCV---for information about how to choose these options based
  on your task and systems specifications, :ref:`The Bottleneck Doctor`
  might be more useful.

Writing image datasets
""""""""""""""""""""""

In most machine learning datasets, images are compressed using ``JPEG`` then
stored. While this scheme is very space-efficient, decoding ``JPEG`` images
requires
significant resources and is usually the bottleneck for loading speed.
Given access to fast
storage (RAM, SSD) in sufficient quantities, other alternatives might be
preferable (see :ref:`The Bottleneck Doctor` for more details).

For the rest of this guide, we'll assume you've already read
:ref:`Writing a dataset to FFCV format`, so you're familiar with the
:class:`ffcv.fields.Field` classes as well as
:class:`ffcv.writer.DatasetWriter`.

Images are supported in FFCV via the :class:`ffcv.fields.RGBImageField` class.
The first initialization parameter of the :class:`~ffcv.fields.RGBImageField` is
the ``write_mode`` argument, which specifies the format with which to write the
dataset, and can take the following values:

- ``jpg``: All the images in the dataset will be stored in JPEG (compressed)
  format.

  .. warning::

    JPEG is a lossy file format. The images read from the data loader might
    be slightly different from the ones passed to the :class:`~ffcv.writer.DatasetWriter`

- ``raw``: All images are stored uncompressed. This dramatically reduces CPU
  usage at loading time but also requires more storage.
  Given enough `RAM` to cache the entirety
  of the dataset, this will usually yield the best performance.
- ``proportion``: This will generate a hybrid dataset with a mix of ``JPEG`` and
  ``raw`` images. Each image will be compressed with probability
  ``compress_probability``. This option is mostly useful for users who wish to
  achieve storage/speed trade-offs between ``jpg`` and ``raw``.
- ``smart``: This is similar to ``proportion`` except that an image will be compressed
  if its ``raw`` representation has area (H x W) more than
  ``smart_threshold``. This option is suited for datasets with
  large variation in image sizes, as it will ensure that a few large outliers do
  not significantly impact the total dataset size or loading speed.

Next, :class:`~ffcv.writer.DatasetWriter` supports a ``jpeg_quality`` argument which
selects the image quality for images that are JPEG-compressed (this
applies to all values ``write_mode`` other than ``raw``). Reducing JPEG quality
will both reduce the size of the file generated and make data loading faster.

Datasets like `ImageNet <http://image-net.org>`_ contain images of various sizes.
For many applications, storing full-sized images is unnecessary, and it may be
beneficial to resize the largest images.
The ``max_resolution`` argument in the initializer of
:class:`~ffcv.writer.DatasetWriter` lets you pick an image side length threshold
for which all larger images are resized (while preserving their aspect ratio).

The following code block provides an example of a
:class:`~ffcv.writer.DatasetWriter` for image data:

.. code-block:: python

    writer = DatasetWriter('my_file.beton', {
            # Roughly 25% of the images will be stored in raw and the other in jpeg
            'image': RGBImageField(
                write_mode='proportion', # Randomly compress
                compress_probability=0.25, # Compress a random 1/4 of the dataset
                max_resolution=(256, 256), # Resize anything above 256 to 256
                jpeg_quality=50  # Use 50% quality when compressing an image using JPG
            ),
            'label': IntField()
        },
    )


Decoding options
'''''''''''''''''

Other fields offer a single :class:`Decoder` suited to read data from the dataset file. For images
we currently offer the following options:

- :class:`~ffcv.fields.decoders.SimpleRGBImageDecoder`: This is the default decoder used when no
  pipeline is passed to the :class:`Loader`. It simply produces the entire image
  and forwards it to the next operations in the pipeline. Note that as a result,
  for this decoder to work all images in a dataset need to have the same
  resolution as they have to fit in the same batch.
- :class:`~ffcv.fields.decoders.RandomResizedCropRGBImageDecoder`. This decoder will first take a
  random section of the image and resize it before populating the batch with
  the image. This decoder is intended to mimic the behavior of ``torchvision.transforms.RandomResizedCrop``.
- :class:`~ffcv.fields.decoders.CenterCropRGBImageDecoder`. Similar to
  :class:`~ffcv.fields.decoders.RandomResizedCropRGBImageDecoder` except that it mimics ``torchvision.transforms.CenterCrop``.

.. code-block:: python

    writer = Loader('my_file.beton',
        batch_size=15,
        num_workers=10
        pipelines = {
            'image': [RandomResizedCropRGBImageDecoder((224, 224))]
            'other_image_field': [CenterCropRGBImageDecoder((224, 224), 224/256)]
        }
    )

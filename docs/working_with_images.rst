Working with Image Data in FFCV
===============================

Images is often responsible for the majority of resources consumed by dataset
(storage and or compute). To cater to the specific needs of each project and hardware,
FFCV offers a wide range of options to control the storage and retrieval of images.


Storage settings
#################

In most machine learning datsets, images are stored compressed using ``JPEG``.
While this is very space efficient, decoding ``JPEG`` images requires
significant resources and is usually the bottleneck. Given access to fast
storage (RAM, SSD) in sufficient quantities, other alternatives might be
preferable (see Benchmarks).

:class:`SimpleRGBImageDecoder` expose the parameter ``write_mode`` which can
take the following values.

- ``jpg``: All the images in the dataset will be stored in ``JPEG``. **Note:**
  ``JPEG`` is a lossy file format. The images read from the data loader might
  be slightly different from the ones passed to the :class:`DatasetWriter`
- ``raw``: All images are stored uncompressed. This dramatically reduces CPU
  usage but will require more storage. Given enough `RAM` to cache the entirety
  of the dataset, this will usually yield the best performance.
- ``proportion``: This will generate a hybrid dataset with some ``JPEG`` and
  ``raw`` images. An image will be compressed with probability
  ``compress_probability``. This option is mostly useful for users who wish to
  achieve storage/speed trade-offs in between ``jpg`` and ``raw``.
- ``smart``: Similar to ``proportion`` except that an image will be compressed
  if if its ``raw`` representation is more than ``smart_threshold`` times
  bigger than it would using ``jpg``. This option is suited for dataset with
  large varation in image sizes as it ensure that couple outliers do not
  significantly impact the total size.

:class:`DatasetWriter` also supports an extra argument `jpeg_quality` which
selects the image quality for images that are stored using ``jpeg``. This
applies to all ``write_mode`` other than ``raw``. It's important to stress
that on top of reducing the size of the file generated, lower image quality also
makes data loading faster.


Datasets like imagenet contain images of various sizes. Depending on the model or
data augmentation pipeline used, it might be benficial to reize he largest images.
:class:`DatasetWriter` let you pick the maximum resolution. All images above above
that threshold will be resized (while preseving their aspect ratio).


.. code-block:: python

    writer = DatasetWriter(num_samples, 'my_file.beton', {
            # Roughly 50% of the images will be stored in raw and the other in jpeg
            'image': SimpleRGBImageDecoder(
                proportion',
                compress_probability=0.5,
                max_resolution=(256, 256),
                jpeg_quality=50  # Use 50% quality when compressing an image using JPG
            ),
            'label': IntField()
        },
    )


Decoding options
'''''''''''''''''

Other fields offer a single :class:`Decoder` suited to read data from the dataset file. For images
we currently offer the following options:

- :class:`SimpleRGBImageDecoder`: This is the default decoder used when no
  pipeline is passed to the :class:`Loader`. It simply produce the entire image
  and forward it to the next operations in the pipeline. Note that as a result
  for this decoder to work all images in a dataset need to have the same
  resolution as they have to fit in the same batch
- :class:`RandomResizedCropRGBImageDecoder`. This decoder will first take a
  random section of the image and resize it before populating the batch with
  the image. This decoder is intended to mimic the behavior of (REF torchvision
  RRC)
- :class:`CenterCropRGBImageDecoder`. Similar to
  :class:`RandomResizedCropRGBImageDecoder` except that it mimics (ref pytorch
  center crop)

.. code-block:: python

    writer = Loader('my_file.beton',
        batch_size=15,
        num_workers=10
        pipelines = {
            'image': [RandomResizedCropRGBImageDecoder((224, 224))]
            'other_image_field': [CenterCropRGBImageDecoder((224, 224), 224/256)]
        }
    )

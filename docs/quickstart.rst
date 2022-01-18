Quickstart
===========

Accelerate *any* learning system with `ffcv`: getting started takes just a few
lines of code!
First, convert your dataset into `ffcv` format (`ffcv` converts both indexed
PyTorch datasets and `WebDatasets <https://github.com/webdataset/webdataset>`_):

.. code-block:: python

    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    my_dataset = make_my_dataset()
    write_path = '/output/path/for/converted/ds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField({
            max_resolution=256,
            jpeg_quality=jpeg_quality
        }),
        'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(ds)

Then replace your old loader with the `ffcv` loader at train time (in PyTorch,
no other changes required!):

.. code-block:: python

    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
    from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder

    # Random resized crop
    decoder = RandomResizedCropRGBImageDecoder((224, 224))

    # Data decoding and augmentation
    image_pipeline = [decoder, Cutout(), ToTensor(), ToTorchImage(), ToDevice(0)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=bs, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines=pipelines)

    # rest of training / validation proceeds identically
    for epoch in range(epochs):
        ...

See :ref:`here <Getting started>` for a more detailed guide to deploying `ffcv` for your dataset.
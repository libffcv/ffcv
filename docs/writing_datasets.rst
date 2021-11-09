Writing a dataset to FFCV format
================================

Datasets in FFCV are stored in a custom ``.beton`` format that allows for fast 
reading (see the _`making_dataloaders` section). 
We provide a utility function :meth:`ffcv.datasets.` for converting PyTorch datasets to
`.beton` files, as well as a conversion script in :file:`scripts/TODO`.

.. code-block:: python 

    from ffcv.writer import DatasetWriter

To write a dataset, we first create a :class:`DatasetWriter`. The writer is
initialized with a dataset size, a path (where the `.beton` will be written),
and a dictionary of *fields*

- :class:`RGBImageField`: chungus

- :class:`BasicField` s: chungus

- :class:`ArrayField`: chungus

.. code-block:: python 

    writer = DatasetWriter(len(my_dataset), write_path, {
        'image': RGBImageField(write_mode='smart', 
                               max_resolution=max_resolution, 
                               smart_threshold=smart_threshold),
        'label': IntField(),
    })

The only remaining step is to write the dataset:

.. code-block:: python

    with writer:
        writer.write_pytorch_dataset(my_dataset,
                                     num_workers=num_workers, 
                                     chunksize=chunk_size)
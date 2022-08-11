Troubleshooting
====================

JSONField
--------------

The current JSONField implementation works best for data samples
formed as a Python dict and containing only non-ascii characters.
When data samples outside of this specification are written to an ffcv dataset,
calls to ``JSONField.unpack()`` after loading may possibly raise various ``JSONDecodeError``s or ``UnicodeDecodeError``s.

To avoid these errors, encapsulate JSONField-intended data such as raw strings inside Python dicts before writing,
and also convert any strings to non-ascii character-only ``utf-8`` encoding, for example via the following string conversion:

.. code-block:: python

    raw_string = 'YOUR STRING HERE'
    to_jsonfield = raw_string.encode('ascii', 'ignore').decode('utf-8')


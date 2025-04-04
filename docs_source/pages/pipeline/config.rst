.. _config:

Config Files 
============

The configuration file is a ``.yml`` file which specifies all of the parameters for each of the methods chosen in a specific SPARCSpy Project run. 

.. code:: yaml
    :caption: Example Configuration File

    ---
    name: "Cellpose Segmentation"
    input_channels: 3
    output_channels: 5
    CytosolSegmentationCellpose:
        input_channels: 2
        output_masks: 2
        shard_size: 120000000 # maxmimum number of pixel per tile
        overlap_px: 100
        nGPUs: 1
        chunk_size: 50 # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        threads: 1 # number of shards / tiles segmented at the same size. should be adapted to the maximum amount allowed by memory.
        cache: "."
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999
        median_filter_size: 6 # Size in pixels
        nucleus_segmentation:
            model: "nuclei"
        cytosol_segmentation:
            model: "cyto2"
        chunk_size: 50
        filtering_threshold: 0.95
    HDF5CellExtraction:
        compression: True
        threads: 80 # threads used in multithreading
        image_size: 400 # image size in pixel
        normalization_range: None #turn of percentile normalization for cells -> otherwise normalise out differences for the alexa647 channel
        cache: "."
        hdf5_rdcc_nbytes: 5242880000 # 5gb 1024 * 1024 * 5000 
        hdf5_rdcc_w0: 1
        hdf5_rdcc_nslots: 50000
    CellFeaturizer:
        channel_classification: 4
        batch_size: 900
        dataloader_worker: 0 #needs to be 0 if using cpu
        inference_device: "cpu"
        screen_label: "Ch3_Featurization"
    LMDSelection:
        processes: 20
        segmentation_channel: 0
        shape_dilation: 16
        smoothing_filter_size: 25
        poly_compression_factor: 30
        path_optimization: "hilbert"
        greedy_k: 15
        hilbert_p: 7

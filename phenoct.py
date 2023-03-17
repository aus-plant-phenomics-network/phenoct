import colorsys
import struct

import cv2
import napari
import numpy as np
import tifffile
from napari_animation import Animation
from plantcv import plantcv as pcv
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm


def create_animation(data, filename):
    """
    Creates a 3D "fly-around" animation and saves to mp4 file.
    :param data: 3D Object
    :param filename: Filename to output
    :return:
    """
    viewer = napari.view_image(data)
    viewer.window.resize(200, 600)

    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 0.25
    viewer.layers['data'].colormap = 'PiYG'
    viewer.camera.angles = (180.0, 0.0, 0.0)

    animation = Animation(viewer)
    viewer.update_console({'animation': animation})

    num_steps = 6
    for i in range(0, num_steps + 1):
        angle = 360.0 * i / num_steps
        viewer.camera.angles = (180.0, angle, 0.0)
        animation.capture_keyframe()
    animation.animate(filename, canvas_only=True)

    viewer.close()


def read_rek_file(filename):
    """
    Reads a REK file into memory.
    :param filename: The location of the .REK file.
    :return: The REK file as a numpy arrray.
    """
    with open(filename, mode="rb") as file:
        hdr_bytes = file.read(2 * 1024)
        hdr = np.frombuffer(hdr_bytes, dtype=np.uint16)

        shape = (hdr[3], hdr[1], hdr[0])

    data = np.memmap(
        filename,
        offset=2048,
        dtype='uint16',
        shape=shape,
        mode='r',
    )

    return data


def write_tiff(data, filename, compression=False):
    """
    Writes a 3D numpy array to a tiff file.
    :param data: 3D numpy array
    :param filename: output filename, optionally including path
    :param compression: boolean True or False, to use zlib compression.
    :return: None
    """
    tifffile.imwrite(
        filename,
        data,
        metadata={'axes': 'ZYX'},
        compression='zlib' if compression else None
    )


def write_8bit_tiff(data, filename, compression=False):
    """
    Converts a 3D numpy array to 8 bit, then write to a tiff file.
    :param data: 3D numpy array
    :param filename: output filename, optionally including path.
    :param compression: boolean True or False, to use zlib compression.
    :return: None
    """
    write_tiff(convert_to_8bit(data), filename, compression)


def convert_to_8bit(data):
    """
    Convert 16bit array to 8bit by down-sampling.
    :param data: 16bit numpy array.
    :return:
    """
    data_8 = (data / 256).astype('uint8')
    return data_8


def segment_sample_holder(data, start_slice=0, stop_slice=None, tube_r=160, tube_thickness=30,
                          attenuation_threshold=None,
                          debug=False, pcv_debug=False):
    """
    Masks the sample from the sample holder by removing the tube and optionally the husks, based on attenuation values.
    :param data: 3D numpy array
    :param start_slice: starting slice (Z direction is negative. i.e. slice 0 is the top.)
    :param stop_slice: slice at which to stop. 2640 is normally a good value.
    :param tube_r: the radius of the sample_holder, in voxels.
    :param tube_thickness: the thickness of the sample_holder tube, in voxels.
    :param attenuation_threshold: the attenuation value above which to include. If not specified, defaults to the mean.
    :param debug: use plot as output for images.
    :param pcv_debug: enable plantcv debugging.
    :return: 3D masked data, and 3D binary masks
    """

    def segment_slice(tube_slice):
        """
        Segments a single vertical slice.
        :param tube_slice: slice index.
        :return: 2D mask
        """
        import matplotlib.pyplot as plt

        if pcv_debug:
            pcv.params.debug = 'plot'
        else:
            pcv.params.debug = None

        med_v = (tube_slice.max() + tube_slice.min()) // 2 if attenuation_threshold is None else attenuation_threshold

        s_thresh = pcv.threshold.binary(gray_img=tube_slice, threshold=med_v, max_value=2 ** 16, object_type='light')
        s_thresh = s_thresh.astype('uint8')

        if debug:
            plt.imshow(s_thresh)
            plt.title("Thresh")
            plt.show()

        h, w = tube_slice.shape

        if debug:
            plt.imshow(tube_slice)
            plt.title("Slice")
            plt.show()

        tube_slice_8bit = (tube_slice // 256).astype('uint8')

        circles = cv2.HoughCircles(tube_slice_8bit, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=150,
                                   maxRadius=0)

        if circles is not None and len(circles) == 1:
            circles = np.round(circles[0, :]).astype("int")
            (x, y, r) = circles[0]
            # Ignore the R from circle finding. Important thing is the centre point.
            circ_mask = np.zeros(tube_slice.shape, dtype=np.uint8)
            cv2.circle(circ_mask, (x, y), tube_r - tube_thickness, 255, 1)

            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(circ_mask, flood_mask, (h // 2, w // 2), 255)

            if debug:
                plt.imshow(circ_mask)
                plt.title("Tube Mask")
                plt.show()
        else:
            plt.imshow(tube_slice)
            plt.title("Slice")
            plt.show()
            print('no circles found. issue.')
            raise

        comb_mask = pcv.logical_and(bin_img1=circ_mask, bin_img2=s_thresh)

        try:
            # if nothing is detected, fill will fail.
            final_mask = pcv.fill(bin_img=comb_mask, size=10)
        except RuntimeError:
            final_mask = comb_mask

        return final_mask

    o_height = stop_slice - start_slice if stop_slice is not None else data.shape[0]
    segmented_data = np.zeros((o_height, data.shape[1], data.shape[2]), dtype='uint16')
    masks = np.zeros((o_height, data.shape[1], data.shape[2]), dtype='uint16')

    for v_slice in (pbar := tqdm(range(0, o_height), total=o_height)):
        pbar.set_description(f"Segmenting slice: {v_slice}")
        img = data[start_slice + v_slice, :, :]

        mask = segment_slice(img)
        masks[v_slice] = mask

        masked = img.copy()
        masked[np.where(mask == 0)] = 0  # pcv.apply_mask(img=img, mask=mask, mask_color='white')
        segmented_data[v_slice] = masked.reshape(img.shape)

    return segmented_data, masks


def auto_crop_sample_holder(masked_data, masks):
    """
    Automatically vertically crops the sample based on masks.
    :param masked_data: 3D numpy array, masked data.
    :param masks: 3D nump array, masks.
    :return:
    """
    # TODO: Crop all empty slices from top, then find a run of ~50 empty slices to determine bottom
    return


def watershed_seeds(masked_data):
    """
    Generates a 3D numpy array of labels/indices based on watershed analysis.
    NOTE, this process may be slow.
    :param masked_data: 3D numpy array
    :return: 3D numpy array of labels.
    """
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(masked_data)
    coords = peak_local_max(distance, labels=masked_data)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=masked_data)
    return labels


def write_colourised_tiff(labelled_data, filename):
    """
    Output labelled data as colourised, composite tiff.
    :param labelled_data: 3D numpy array, labelled from watershed.
    :param filename: output filename, optionally including path
    :return: None
    """

    def imagej_metadata_tags(metadata, byteorder):
        """
        Implemented from : https://stackoverflow.com/a/50263336/5447556

        Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

        The tags can be passed to the TiffWriter save function as extratags.

        :param metadata: metadata dictionary
        :param byteorder: byte order
        :return: encoded metadata
        """
        header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
        bytecounts = [0]
        body = []

        def writestring(_data, _byteorder):
            return _data.encode('utf-16' + {'>': 'be', '<': 'le'}[_byteorder])

        def writedoubles(_data, _byteorder):
            return struct.pack(_byteorder + ('d' * len(data)), *_data)

        def writebytes(_data, _byteorder):
            return _data.tobytes()

        metadata_types = (
            ('Info', b'info', 1, writestring),
            ('Labels', b'labl', None, writestring),
            ('Ranges', b'rang', 1, writedoubles),
            ('LUTs', b'luts', None, writebytes),
            ('Plot', b'plot', 1, writebytes),
            ('ROI', b'roi ', 1, writebytes),
            ('Overlays', b'over', None, writebytes))

        for key, mtype, count, func in metadata_types:
            if key not in metadata:
                continue
            if byteorder == '<':
                mtype = mtype[::-1]
            values = metadata[key]
            if count is None:
                count = len(values)
            else:
                values = [values]
            header.append(mtype + struct.pack(byteorder + 'I', count))
            for value in values:
                data = func(value, byteorder)
                body.append(data)
                bytecounts.append(len(data))

        body = b''.join(body)
        header = b''.join(header)
        data = header + body
        bytecounts[0] = len(header)
        bytecounts = struct.pack(byteorder + ('I' * len(bytecounts)), *bytecounts)
        return ((50839, 'B', len(data), data, True),
                (50838, 'I', len(bytecounts) // 4, bytecounts, True))

    N = np.max(labelled_data)
    HSV_tuples = [(x * 1.0 / N, 1.0, 1.0) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    RGBK = dict(zip(range(1, N + 2), RGB_tuples))
    RGBK[0] = (0.0, 0.0, 0.0)

    for k, v in RGBK.items():
        RGBK[k] = [int(x * 255) for x in v]

    def get_r(x):
        return RGBK[x][0]

    def get_g(x):
        return RGBK[x][1]

    def get_b(x):
        return RGBK[x][2]

    get_r_v = np.vectorize(get_r)
    get_g_v = np.vectorize(get_g)
    get_b_v = np.vectorize(get_b)

    r = get_r_v(labelled_data)
    g = get_g_v(labelled_data)
    b = get_b_v(labelled_data)

    coloured = np.stack([r, g, b], axis=1)

    val_range = np.arange(256, dtype=np.uint8)

    lut_red = np.zeros((3, 256), dtype=np.uint8)
    lut_red[0, :] = val_range
    lut_green = np.zeros((3, 256), dtype=np.uint8)
    lut_green[1, :] = val_range
    lut_blue = np.zeros((3, 256), dtype=np.uint8)
    lut_blue[2, :] = val_range

    ijmeta = imagej_metadata_tags({'LUTs': [lut_red, lut_green, lut_blue]}, '>')

    tifffile.imwrite(
        filename,
        coloured.astype('uint8'),
        imagej=True,
        extratags=ijmeta,
        metadata={'axes': 'ZCYX',
                  # mode: composite breaks SyGlass but is necessary for ImageJ
                  'mode': 'composite'
                  },
        compression='zlib'
    )


def crop_segmented(s_data):
    """
    Finds the non-zero extents of a 3D array and copies the original by slicing it.
    :param s_data:
    :return: 3D numpy array
    """

    # get the indices of all non-zero values in all three dimensions
    nonzero_indices = np.nonzero(s_data)

    # create a new array with only the non-zero values
    cropped_arr = s_data[nonzero_indices[0].min():nonzero_indices[0].max() + 1,
                  nonzero_indices[1].min():nonzero_indices[1].max() + 1,
                  nonzero_indices[2].min():nonzero_indices[2].max() + 1]

    return cropped_arr.copy()

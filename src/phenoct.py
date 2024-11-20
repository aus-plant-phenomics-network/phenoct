import colorsys
import struct

import cv2
import time
import napari
import numpy as np
import tifffile
from napari_animation import Animation
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

from tqdm import tqdm
import matplotlib.pyplot as plt


class CT:
    data = None
    segmented_data = None
    labels = None

    def __init__(self, filename):
        self.filename = filename

        self.read_rek_file(filename)

    def read_rek_file(self, filename):
        """
        Reads a REK file into memory.
        :param filename: The location of the .REK file.
        :return: The REK file as a numpy arrray.
        """
        with open(filename, mode="rb") as file:
            hdr_bytes = file.read(2 * 1024)
            hdr = np.frombuffer(hdr_bytes, dtype=np.uint16)

            shape = (hdr[3], hdr[1], hdr[0])

        self.data = np.memmap(
            filename,
            offset=2048,
            dtype="uint16",
            shape=shape,
            mode="r",
        )

        print(self.data.dtype)

    def crop(self):
        """
        Finds the non-zero extents of a 3D array and copies the original by slicing it.
        """

        self.data = crop_any(self.data)

    def downsample(self):
        self.data = (self.data // 256).astype("uint8")

    def crop_segmented(self):
        """
        Finds the non-zero extents of a 3D array and copies the original by slicing it.
        """

        if self.segmented_data is None:
            raise Exception("Not yet segemented.")

        self.segmented_data = crop_any(self.segmented_data)

    def write_data_tiff(self, out_filename, bit_depth=16, compression=True):
        """
        Writes a 3D numpy array to a tiff file.
        :param bit_depth:
        :param out_filename: output filename, optionally including path
        :param compression: boolean True or False, to use zlib compression.
        :return: None
        """

        # Determine the appropriate data type based on the input string
        convert_and_write_tiff(self.data, bit_depth, compression, out_filename)

    def write_segmented_data_tiff(self, out_filename, bit_depth=16, compression=True):
        """
        Writes a 3D numpy array to a tiff file.
        :param bit_depth:
        :param out_filename: output filename, optionally including path
        :param compression: boolean True or False, to use zlib compression.
        :return: None
        """
        if self.segmented_data is None:
            raise Exception("Not yet segemented.")
        # Determine the appropriate data type based on the input string
        convert_and_write_tiff(
            self.segmented_data, bit_depth, compression, out_filename
        )

    def view_data(self):
        viewer = napari.view_image(self.data)


class Tube(CT):

    def create_animation(self, filename, colormap="gray", rendering="attenuated_mip"):
        """
        Creates a 3D "fly-around" animation and saves to mp4 file.
        :param rendering:
        :param colormap:
        :param filename: Filename to output
        :return:
        """

        if self.segmented_data is None:
            raise Exception("Not yet segemented.")

        viewer = napari.view_image(self.segmented_data)

        viewer.window.resize(200, 600)

        viewer.dims.ndisplay = 3
        viewer.camera.zoom = 0.25
        viewer.layers[0].colormap = colormap
        viewer.layers[0].rendering = rendering
        viewer.camera.angles = (180.0, 0.0, 0.0)

        animation = Animation(viewer)
        viewer.update_console({"animation": animation})

        num_steps = 6
        for i in range(0, num_steps + 1):
            angle = 360.0 * i / num_steps
            viewer.camera.angles = (180.0, angle, 0.0)
            animation.capture_keyframe()
        animation.animate(filename, canvas_only=True)

        # TODO: Ensure viewer closes properly
        viewer.close()

    def segment_sample_holder(
        self,
        start_slice=0,
        stop_slice=None,
        tube_r=160,
        tube_thickness=30,
        attenuation_threshold=None,
        debug=False,
    ):
        """
        Masks the sample from the sample holder by removing the tube and optionally the husks, based on attenuation values.
        :param data: 3D numpy array
        :param start_slice: starting slice (Z direction is negative. i.e. slice 0 is the top.)
        :param stop_slice: slice at which to stop. 2640 is normally a good value.
        :param tube_r: the radius of the sample_holder, in voxels.
        :param tube_thickness: the thickness of the sample_holder tube, in voxels.
        :param attenuation_threshold: either the attenuation value above which to include (int), or a list of 2 ints.
                If not specified, defaults to the mean.
        :param debug: use plot as output for images.
        :param pcv_debug: enable plantcv debugging.
        :return: 3D masked data, and 3D binary masks
        """

        def segment_slice(_v_slice):
            """
            Segments a single vertical slice.
            :param _v_slice: slice index.
            :return: 2D mask
            """

            if isinstance(attenuation_threshold, int):

                min_v = attenuation_threshold
                _, s_thresh = cv2.threshold(_v_slice, min_v, 2**16, cv2.THRESH_BINARY)

            elif isinstance(attenuation_threshold, list) or isinstance(
                attenuation_threshold, tuple
            ):
                min_v = attenuation_threshold[0]
                max_v = attenuation_threshold[1]

                _, s_thresh_min = cv2.threshold(
                    _v_slice, min_v, 2**16, cv2.THRESH_BINARY
                )

                _, s_thresh_max = cv2.threshold(
                    _v_slice, max_v, 2**16, cv2.THRESH_BINARY_INV
                )

                s_thresh = cv2.bitwise_and(s_thresh_min, s_thresh_max)

            elif attenuation_threshold is None:

                min_v = (_v_slice.max() + _v_slice.min()) // 2

                _, s_thresh = cv2.threshold(_v_slice, min_v, 2**16, cv2.THRESH_BINARY)

            else:
                raise Exception(
                    "Please specify attenuation threshold as an integer or list."
                )

            s_thresh = s_thresh.astype("uint8")

            if debug:
                plt.imshow(s_thresh)
                plt.title("Thresh")
                plt.show()

            h, w = _v_slice.shape

            if debug:
                plt.imshow(_v_slice)
                plt.title("Slice")
                plt.show()

            tube_slice_8bit = (_v_slice // 256).astype("uint8")

            circles = cv2.HoughCircles(
                tube_slice_8bit,
                cv2.HOUGH_GRADIENT,
                1,
                200,
                param1=50,
                param2=30,
                minRadius=150,
                maxRadius=0,
            )

            if circles is not None and len(circles) == 1:
                circles = np.round(circles[0, :]).astype("int")
                (x, y, r) = circles[0]
                # Ignore the R from circle finding. Important thing is the centre point.
                circ_mask = np.zeros(_v_slice.shape, dtype=np.uint8)
                cv2.circle(circ_mask, (x, y), tube_r - tube_thickness, 255, 1)

                flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(circ_mask, flood_mask, (h // 2, w // 2), 255)

                if debug:
                    plt.imshow(circ_mask)
                    plt.title("Tube Mask")
                    plt.show()
            else:
                plt.imshow(_v_slice)
                plt.title("Slice")
                plt.show()
                print("no circles found. issue.")
                raise

            comb_mask = cv2.bitwise_and(circ_mask, s_thresh)

            try:
                # if nothing is detected, fill will fail.
                # Find and fill contours
                bool_img = remove_small_objects(comb_mask.astype(bool), 10)

                # Cast boolean image to binary and make a copy of the binary image for returning
                final_mask = np.copy(bool_img.astype(np.uint8) * 255)

            except RuntimeError:
                final_mask = comb_mask

            return final_mask

        # o_height = stop_slice - start_slice if stop_slice is not None else self.data.shape[0]
        segmented_data = np.zeros(self.data.shape, dtype="uint16")
        masks = np.zeros(self.data.shape, dtype="uint16")

        if stop_slice is None:
            stop_slice = self.data.shape[0]

        for v_slice in (
            pbar := tqdm(range(start_slice, stop_slice), total=stop_slice - start_slice)
        ):
            pbar.set_description(f"Segmenting slice: {v_slice}")
            img = self.data[v_slice, :, :]

            mask = segment_slice(img)
            masks[v_slice] = mask

            masked = img.copy()
            masked[np.where(mask == 0)] = (
                0  # pcv.apply_mask(img=img, mask=mask, mask_color='white')
            )
            segmented_data[v_slice] = masked.reshape(img.shape)

        self.segmented_data = segmented_data

    def watershed_seeds(self):
        """
        Generates a 3D numpy array of labels/indices based on watershed analysis.
        NOTE, this process may be slow.
        :return: 3D numpy array of labels.
        """
        if self.segmented_data is None:
            raise Exception("Data has not yet been segmented.")

        data_to_watershed, t = crop_any(self.segmented_data, return_translations=True)

        # print(translations)
        # return

        # Now we want to separate the two objects in image
        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(data_to_watershed)
        coords = peak_local_max(distance, labels=data_to_watershed)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=data_to_watershed)
        translated_labels = np.zeros(self.segmented_data.shape, dtype=labels.dtype)
        np.copyto(translated_labels[tuple([slice(0, n) for n in labels.shape])], labels)
        for i in range(3):
            translated_labels = np.roll(translated_labels, t[i], axis=i)
        self.labels = translated_labels

    def write_colourised_tiff(self, filename):
        """
        Output labelled data as colourised, composite tiff.
        :param filename: output filename, optionally including path
        :return: None
        """

        if self.labels is None:
            raise Exception("Not yet watershed.")

        def imagej_metadata_tags(metadata, byteorder):
            """
            Implemented from : https://stackoverflow.com/a/50263336/5447556

            Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

            The tags can be passed to the TiffWriter save function as extratags.

            :param metadata: metadata dictionary
            :param byteorder: byte order
            :return: encoded metadata
            """
            header = [{">": b"IJIJ", "<": b"JIJI"}[byteorder]]
            bytecounts = [0]
            body = []

            def writestring(_data, _byteorder):
                return _data.encode("utf-16" + {">": "be", "<": "le"}[_byteorder])

            def writedoubles(_data, _byteorder):
                return struct.pack(_byteorder + ("d" * len(data)), *_data)

            def writebytes(_data, _byteorder):
                return _data.tobytes()

            metadata_types = (
                ("Info", b"info", 1, writestring),
                ("Labels", b"labl", None, writestring),
                ("Ranges", b"rang", 1, writedoubles),
                ("LUTs", b"luts", None, writebytes),
                ("Plot", b"plot", 1, writebytes),
                ("ROI", b"roi ", 1, writebytes),
                ("Overlays", b"over", None, writebytes),
            )

            for key, mtype, count, func in metadata_types:
                if key not in metadata:
                    continue
                if byteorder == "<":
                    mtype = mtype[::-1]
                values = metadata[key]
                if count is None:
                    count = len(values)
                else:
                    values = [values]
                header.append(mtype + struct.pack(byteorder + "I", count))
                for value in values:
                    data = func(value, byteorder)
                    body.append(data)
                    bytecounts.append(len(data))

            body = b"".join(body)
            header = b"".join(header)
            data = header + body
            bytecounts[0] = len(header)
            bytecounts = struct.pack(byteorder + ("I" * len(bytecounts)), *bytecounts)
            return (
                (50839, "B", len(data), data, True),
                (50838, "I", len(bytecounts) // 4, bytecounts, True),
            )

        N = np.max(self.labels)
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

        r = get_r_v(self.labels)
        g = get_g_v(self.labels)
        b = get_b_v(self.labels)

        coloured = np.stack([r, g, b], axis=1)

        val_range = np.arange(256, dtype=np.uint8)

        lut_red = np.zeros((3, 256), dtype=np.uint8)
        lut_red[0, :] = val_range
        lut_green = np.zeros((3, 256), dtype=np.uint8)
        lut_green[1, :] = val_range
        lut_blue = np.zeros((3, 256), dtype=np.uint8)
        lut_blue[2, :] = val_range

        ijmeta = imagej_metadata_tags({"LUTs": [lut_red, lut_green, lut_blue]}, ">")

        tifffile.imwrite(
            filename,
            coloured.astype("uint8"),
            imagej=True,
            extratags=ijmeta,
            metadata={
                "axes": "ZCYX",
                # mode: composite breaks SyGlass but is necessary for ImageJ
                "mode": "composite",
            },
            compression="zlib",
        )

    def view_segmented_data(self, contrast_limits=(0, 15000)):
        if self.segmented_data is None:
            raise Exception("Data has not yet been segmented.")
        viewer = napari.view_image(self.segmented_data, contrast_limits=contrast_limits)

    def view_labelled_data(self):
        viewer = napari.view_image(self.data)

        if self.labels is None:
            raise "Not yet watershed."
        labels_layer = viewer.add_labels(self.labels, name="segmentation")


def crop_any(data, return_translations=False):
    """

    :param return_translations:
    :param data:
    :return:
    """
    nonzero_indices = np.nonzero(data)
    print(nonzero_indices)

    # create a new array with only the non-zero values
    cropped_arr = data[
        nonzero_indices[0].min() : nonzero_indices[0].max() + 1,
        nonzero_indices[1].min() : nonzero_indices[1].max() + 1,
        nonzero_indices[2].min() : nonzero_indices[2].max() + 1,
    ]
    if return_translations:
        return cropped_arr, (
            nonzero_indices[0].min(),
            nonzero_indices[1].min(),
            nonzero_indices[2].min(),
        )
    else:
        return cropped_arr


def convert_and_write_tiff(data, bit_depth, compression, out_filename):
    if bit_depth == 8:
        data_type = np.int8
        conversion_factor = 256
        converted_array = (data // conversion_factor).astype(data_type)
    elif bit_depth == 16:
        converted_array = data
    elif bit_depth == 32:
        data_type = np.uint32
        conversion_factor = 65536
        converted_array = data.astype(data_type)
        converted_array = converted_array * conversion_factor

    else:
        raise ValueError(
            f"Invalid dtype: {bit_depth}. Please choose between 8, 16, or 32."
        )
    #print(data.dtype)
    #print(f"Min value: {np.min(data)}")
    #print(f"Max value: {np.max(data)}")
    #print(converted_array.dtype)
    #print(f"Min value: {np.min(converted_array)}")
    #print(f"Max value: {np.max(converted_array)}")
    estimated_size = converted_array.nbytes
    bigtiff_needed = estimated_size > 4 * 1024**3  # 4GB threshold

    print(f"Compression: {compression}, BigTIFF: {bigtiff_needed}")
    
    tifffile.imwrite(
        out_filename,
        converted_array,
        metadata={"axes": "ZYX"},
        compression="zlib" if compression else None,
        bigtiff=bigtiff_needed
    )

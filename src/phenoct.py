import colorsys
import struct

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm


class CT:

    def __init__(self, filename: str):
        self.filename = filename
        self.data = None
        self.segmented_data = None
        self.labels = None

        self.read_rek_file(filename)

    def __enter__(self):
        # Initialize or allocate resources
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "segmented_data"):
            del self.segmented_data
        if hasattr(self, "labels"):
            del self.labels

    def read_rek_file(self, filename: str):
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
        print(self.data.shape)

    def crop(self):
        """
        Finds the non-zero extents of a 3D array and copies the original by slicing it.
        """

        self.data = crop_any(self.data)

    def downsample(self):
        self.data = (self.data // 256).astype("uint8")

    def write_maximal_projections(
        self, out_filename, compression=True, normalized=True
    ):

        if normalized:
            output_data = (
                (self.segmented_data - self.segmented_data.min())
                / (self.segmented_data.max() - self.segmented_data.min())
                * 255
            ).astype(np.uint8)
        else:
            output_data = self.segmented_data

        for axis in range(1, 3):
            flattened_0 = np.max(output_data, axis=axis)
            tifffile.imwrite(
                f"{out_filename}_{axis}.tiff",
                flattened_0,
                metadata={},
                compression="zlib" if compression else None,
            )

        rotated_data = scipy.ndimage.rotate(
            output_data, 45, axes=(1, 2), reshape=True, order=1
        )

        for axis in range(1, 3):
            rotated_projection = np.max(rotated_data, axis=axis)
            tifffile.imwrite(
                f"{out_filename}_rotated_{axis}.tiff",
                rotated_projection,
                metadata={},
                compression="zlib" if compression else None,
            )

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

    def write_segmented_data_tiff(self, out_filename, bit_depth=16, compression=True, crop=True):
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
        if crop:
            output_data = crop_any(self.segmented_data)
        else:
            output_data = self.segmented_data
        convert_and_write_tiff(
            output_data, bit_depth, compression, out_filename
        )


class Tube(CT):

    def create_animation(self, filename, colormap="viridis"):
        """
        Creates a 3D "fly-around" animation and saves to mp4 file.
        :param colormap:
        :param filename: Filename to output
        :return:
        """

        if self.segmented_data is None:
            raise Exception("Not yet segemented.")

        self.crop_segmented()

        render_vol = self.segmented_data.astype(float)

        # Denoise
        render_vol = gaussian_filter(render_vol, sigma=1.0)

        # Mask background hard
        render_vol[self.segmented_data == 0] = 0

        # Downsample for rendering
        render_vol = render_vol[::2, ::2, ::2]

        normalised_segmented_data = (
            (render_vol - render_vol.min())
            / (render_vol.max() - render_vol.min())
            * 255
        ).astype(np.uint8)

        grid = pv.ImageData()
        grid.dimensions = np.array(normalised_segmented_data.shape)
        grid.spacing = (1, 1, 1)  # Adjust if needed
        grid.point_data["values"] = normalised_segmented_data.flatten(
            order="F"
        )  # Column-major flattening

        # Translate the object to the origin (0, 0, 0)
        translation_vector = [-x for x in grid.center]
        grid.translate(translation_vector)

        # Set up the plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(grid, opacity="linear", cmap=colormap)
        plotter.remove_scalar_bar()
        plotter.window_size = [208, 608]
        plotter.open_movie(filename, framerate=12)
        initial_distance = 3000

        initial_cam_pos = (grid.center[0] // 2, initial_distance, initial_distance)
        focus = (
            grid.center[0] // 2,
            grid.center[1],
            grid.center[2],
        )

        # Animate camera orbit around Z-axis
        n_frames = 36
        for i in range(n_frames):
            angle = np.radians(i * (360 / n_frames))  # Convert degrees to radians
            y = initial_distance * np.cos(angle)  # Move along Y
            z = initial_distance * np.sin(angle)  # Move along Z

            plotter.camera_position = [(initial_cam_pos[0], y, z), focus, (-1, 0, 0)]
            plotter.write_frame()

        plotter.close()


    def segment_sample_holder(
            self,
            start_slice=0,
            stop_slice=None,
            radial_margin=2,
            n_angles=360,
            debug=False,
    ):
        """
        Remove an acrylic tube by detecting its cylindrical shell using
        angularly-adaptive radial geometry.

        This method:
        - does NOT assume tube is brighter than sample
        - does NOT model the interior
        - ignores caps automatically
        - adapts to ellipticity / slight miscentering
        """

        if stop_slice is None:
            stop_slice = self.data.shape[0]

        volume = self.data[start_slice:stop_slice]

        # ------------------------------------------------------------
        # 1. Z-maximum projection (stable tube signal)
        # ------------------------------------------------------------
        proj = np.max(volume, axis=0).astype(np.float32)

        h, w = proj.shape
        yy, xx = np.indices((h, w))

        # ------------------------------------------------------------
        # 2. Robust tube centre (intensity-weighted image moments)
        # ------------------------------------------------------------
        total = proj.sum()
        cy = (yy * proj).sum() / total
        cx = (xx * proj).sum() / total

        # ------------------------------------------------------------
        # 3. Radial + angular coordinates
        # ------------------------------------------------------------
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        theta = np.arctan2(yy - cy, xx - cx)

        max_r = int(rr.max()) - 1

        # ------------------------------------------------------------
        # 4. Global radial profile (for sanity checking)
        # ------------------------------------------------------------
        radial_profile = np.zeros(max_r, dtype=np.float32)

        for r in range(max_r):
            shell = (rr >= r) & (rr < r + 1)
            if np.any(shell):
                radial_profile[r] = np.median(proj[shell])

        radial_profile = gaussian_filter1d(radial_profile, sigma=2)

        grad_global = np.gradient(radial_profile)

        # ------------------------------------------------------------
        # 5. Angularly adaptive inner tube radius
        # ------------------------------------------------------------
        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
        inner_radii = np.zeros(n_angles, dtype=np.float32)

        angle_width = np.pi / n_angles  # ~1 degree

        for i, a in enumerate(angles):
            sector = np.abs(np.angle(np.exp(1j * (theta - a)))) < angle_width

            if not np.any(sector):
                inner_radii[i] = np.nan
                continue

            r_vals = rr[sector]
            i_vals = proj[sector]

            order = np.argsort(r_vals)
            r_sorted = r_vals[order]
            i_sorted = i_vals[order]

            prof = gaussian_filter1d(i_sorted, sigma=2)
            grad = np.gradient(prof)

            r_start = int(0.25 * len(prof))  # ignore centre/sample
            tube_idx = r_start + np.argmax(grad[r_start:])

            baseline = np.median(np.abs(grad[:tube_idx]))
            inner_candidates = np.where(np.abs(grad[:tube_idx]) < baseline)[0]

            if len(inner_candidates):
                inner_radii[i] = r_sorted[inner_candidates[-1]]
            else:
                inner_radii[i] = np.nan

        # clean angular outliers
        med_r = np.nanmedian(inner_radii)
        inner_radii[np.isnan(inner_radii)] = med_r
        inner_radii = gaussian_filter1d(inner_radii, sigma=5)

        if debug:
            print(f"Tube centre: cx={cx:.2f}, cy={cy:.2f}")
            print(f"Median inner radius: {med_r:.2f}")

        # ------------------------------------------------------------
        # 6. Build angularly adaptive tube-removal mask
        # ------------------------------------------------------------
        remove_mask_2d = np.zeros_like(rr, dtype=bool)

        for i, a in enumerate(angles):
            sector = np.abs(np.angle(np.exp(1j * (theta - a)))) < angle_width
            remove_r = inner_radii[i] - radial_margin
            remove_mask_2d[sector & (rr >= remove_r)] = True

        # ------------------------------------------------------------
        # 7. Apply to full volume
        # ------------------------------------------------------------
        segmented_data = self.data.copy()
        masks = np.zeros_like(self.data, dtype=np.uint8)

        for z in tqdm(range(start_slice, stop_slice), desc="Removing tube"):
            segmented_data[z][remove_mask_2d] = 0
            masks[z][~remove_mask_2d] = 255

        # ------------------------------------------------------------
        # 7. Histogram-based background removal (robust)
        # ------------------------------------------------------------
        interior = (masks > 0) & (segmented_data > 0)
        vals = segmented_data[interior]

        hist, bin_edges = np.histogram(vals, bins=256)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

        bg_peak_idx = np.argmax(hist_smooth)
        bg_peak_val = bin_edges[bg_peak_idx]

        cutoff = bg_peak_val * 1.2  # safe margin above background
        print(f"Histogram background peak: {bg_peak_val:.1f}")
        print(f"Cutoff used: {cutoff:.1f}")

        candidate = segmented_data.copy()
        candidate[candidate < cutoff] = 0

        safe_removal = False
        if safe_removal:
            # Remove diffuse noise by connectivity
            labels_cc, n = ndi.label(candidate > 0)
            sizes = ndi.sum(candidate > 0, labels_cc, range(1, n + 1))

            cleaned = np.zeros_like(candidate)
            for i, s in enumerate(sizes, start=1):
                if s > 200:
                    cleaned[labels_cc == i] = candidate[labels_cc == i]

            self.segmented_data = cleaned[start_slice:stop_slice]

        else:

            cleaned = candidate

            self.segmented_data = candidate[start_slice:stop_slice]

        if debug:
            z = (start_slice + stop_slice) // 2

            before = segmented_data[z]
            after = cleaned[z]

            p_lo, p_hi = np.percentile(before[before > 0], (1, 99))
            before_disp = np.clip((before - p_lo) / (p_hi - p_lo), 0, 1)
            after_disp = np.clip((after - p_lo) / (p_hi - p_lo), 0, 1)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(before_disp, cmap="gray")
            ax[0].set_title("Before cleanup")

            ax[1].imshow(after_disp, cmap="gray")
            ax[1].set_title("After cleanup")

            ax[2].imshow((before > 0) & (after == 0), cmap="hot")
            ax[2].set_title("Removed voxels")

            for a in ax:
                a.axis("off")

            plt.show()


        # ------------------------------------------------------------
        # 8. Debug visualisations
        # ------------------------------------------------------------
        if debug:
            # Radial profile + gradient
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].plot(radial_profile)
            ax[0].set_title("Global radial median intensity")
            ax[1].plot(grad_global)
            ax[1].set_title("Global radial gradient")
            plt.show()

            # Inner radius vs angle
            plt.figure(figsize=(8, 3))
            plt.plot(np.rad2deg(angles), inner_radii)
            plt.xlabel("Angle (deg)")
            plt.ylabel("Inner tube radius (px)")
            plt.title("Angular inner tube radius")
            plt.show()

            # Overlay detected boundary on projection
            p_lo, p_hi = np.percentile(proj, (1, 99))
            proj_disp = np.clip((proj - p_lo) / (p_hi - p_lo), 0, 1)

            rgb = np.dstack([proj_disp, proj_disp, proj_disp])

            for i, a in enumerate(angles):
                r0 = inner_radii[i]
                r1 = r0 - radial_margin

                x0 = cx + r0 * np.cos(a)
                y0 = cy + r0 * np.sin(a)
                x1 = cx + r1 * np.cos(a)
                y1 = cy + r1 * np.sin(a)

                if 0 <= int(y0) < h and 0 <= int(x0) < w:
                    rgb[int(y0), int(x0)] = [255, 0, 0]  # red = detected wall

                if 0 <= int(y1) < h and 0 <= int(x1) < w:
                    rgb[int(y1), int(x1)] = [0, 255, 0]  # green = removal boundary

            plt.figure(figsize=(6, 6))

            plt.imshow(rgb)
            plt.title("Red = detected wall, Green = effective removal")
            plt.axis("off")
            plt.show()


    def watershed_seeds(self, debug=False):


        if self.segmented_data is None:
            raise Exception("Data has not yet been segmented.")

        data_to_watershed, t = crop_any(self.segmented_data, return_translations=True)

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
        # ------------------------------------------------------------
        # 5. Debug
        # ------------------------------------------------------------
        if debug and len(coords) > 0:
            z = coords[len(coords) // 2][0]
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(mask[z], cmap="gray");
            ax[0].set_title("Grain mask")
            ax[1].imshow(distance[z], cmap="magma");
            ax[1].set_title("Distance")
            ax[2].imshow(labels[z], cmap="tab20");
            ax[2].set_title("Labels")
            for a in ax: a.axis("off")
            plt.show()

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
    # print(data.dtype)
    # print(f"Min value: {np.min(data)}")
    # print(f"Max value: {np.max(data)}")
    # print(converted_array.dtype)
    # print(f"Min value: {np.min(converted_array)}")
    # print(f"Max value: {np.max(converted_array)}")
    estimated_size = converted_array.nbytes
    bigtiff_needed = estimated_size > 4 * 1024**3  # 4GB threshold

    print(f"Compression: {compression}, BigTIFF: {bigtiff_needed}")

    tifffile.imwrite(
        out_filename,
        converted_array,
        metadata={"axes": "ZYX"},
        compression="zlib" if compression else None,
        bigtiff=bigtiff_needed,
    )

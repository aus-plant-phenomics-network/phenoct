import requests
import os
from tqdm import tqdm
import numpy as np
import pyvista as pv
import phenoct


def fetch_example_tube(filename):
    try:
        url = "https://projects.pawsey.org.au/appf-quick-data-sharing/XRAYCT_V_X-004481-01_220526112956002_ears_1_200mm_084mu_0645-UniSA-SH0017_220601091114171.rek"

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(filename):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(filename, "wb") as file:
                # Create a progress bar
                progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

                progress_bar.close()

            print(f"File downloaded successfully: {filename}")
        else:
            print("File already exists")
    except Exception as e:
        print(f"An error occurred: {e}")


out_dir = "test_outputs"
filename = f"{out_dir}/example_tube.rek"
fetch_example_tube(filename)

tube = phenoct.Tube("test_outputs/example_tube.rek")

tube.segment_sample_holder(start_slice=300, stop_slice=2500, debug=False)

tube.create_animation("ct_animation_test.mp4")


# tube.view_data()
# tube.write_data_tiff("test_outputs/32.tiff", 32)
# # tube.write_data_tiff("16.tiff")
#
# tube.segment_sample_holder(start_slice=300, stop_slice=2500, debug=False)
# # tube.crop_segmented()
# tube.write_segmented_data_tiff("test_outputs/seg16.tiff")
# tube.write_segmented_data_tiff("test_outputs/seg32.tiff", 32)
#
# # Load the image using PIL
# img = Image.open("test_outputs/seg16.tiff")
#
# # Convert the PIL image to a numpy array
# img_arr = np.array(img)
#
# # Print the data type and min/max values
# print(f"Data shape: {img_arr.shape}")
# print(f"Data type: {img_arr.dtype}")
# print(f"Min value: {np.min(img_arr)}")
# print(f"Max value: {np.max(img_arr)}")
#
#
# # higher attenuation threshold and fewer slices for testing. This is slow.
# tube.segment_sample_holder(start_slice=1550, stop_slice=1600, debug=False)
#
# tube.watershed_seeds()
#
# # Uncomment this to open in Napari Viewer.
# # tube.view_segmented_data()
#
# tube.write_colourised_tiff("test_outputs/a_cc.tiff")
#
# tube.create_animation("test_outputs/tube.mp4")

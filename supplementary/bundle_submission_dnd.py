import os
import scipy.io as sio
import numpy as np
import h5py


# ------------------------
# SETTINGS
# ------------------------
submission_folder = "results_DND"
num_images = 50
patches_per_image = 20


# ------------------------
# STEP 1: Bundle function (as given)
# ------------------------
def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    os.makedirs(out_folder, exist_ok=True)

    israw = False
    eval_version = "1.0"

    for i in range(num_images):
        Idenoised = np.zeros((patches_per_image,), dtype=object)
        for bb in range(patches_per_image):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            with h5py.File(os.path.join(submission_folder, filename), 'r') as f:
                patch = f['data'][()]  # load dataset as NumPy array
                patch = np.clip(patch.astype(np.float32) / 255, a_min=0, a_max=1)
                Idenoised[bb] = patch
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


# ------------------------
# STEP 2: Run bundling
# ------------------------
print("Bundling patches into submission-ready .mat files...")
bundle_submissions_srgb(submission_folder)
print("Bundling complete! Submission files are in:", os.path.join(submission_folder, "bundled"))

import logging
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from deepspine import DeepSpine
import io
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_deepspine_pipeline(scan_volume, pixel_spacing, slice_thickness):
    spnt = DeepSpine(verbose=False)

    logger.info(f'Scan has {scan_volume.shape[-1]} sagittal slices, of dimension {scan_volume.shape[0]}x{scan_volume.shape[1]} ({pixel_spacing} mm pixel spacing) and {slice_thickness} mm slice thickness.')

    if len(scan_volume.shape) > 3:
        scan_volume = np.squeeze(scan_volume[:,:,[0], :])  # Убираем лишнюю ось

    vert_dicts = spnt.detect_vb(scan_volume, pixel_spacing)
    logger.info(f'{len(vert_dicts)} vertebrae detected: {[v["predicted_label"] for v in vert_dicts]}')

    images_with_masks = []

    for slice_idx in range(scan_volume.shape[-1]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(scan_volume[:,:,slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx+1}')
        ax.axis('off')
        for vert_dict in vert_dicts:
            if slice_idx in vert_dict['slice_nos']:
                poly_idx = int(vert_dict['slice_nos'].index(slice_idx))
                poly = np.array(vert_dict['polys'][poly_idx])
                ax.add_patch(Polygon(poly, ec='y', fc='none'))
                ax.text(np.mean(poly[:,0]), np.mean(poly[:,1]), vert_dict['predicted_label'], c='y', ha='center', va='center')

        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        images_with_masks.append(Image.open(buf).convert('RGB'))
        plt.close(fig)

    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan_volume)
    ivd_grades = spnt.grade_ivds(ivd_dicts)

    return images_with_masks, ivd_grades

if __name__ == "__main__":
    from dicom_io import load_dicoms_from_folder
    scan = load_dicoms_from_folder(r"F:\WorkSpace\Z-Union\test MRI", require_extensions=True)
    run_deepspine_pipeline(scan.volume, scan.pixel_spacing, scan.slice_thickness)
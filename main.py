import logging
import numpy as np

from deepspine import DeepSpine

from PIL import Image
import cv2

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
        # Извлекаем срез и нормализуем в 8-битное изображение
        slice_img = scan_volume[:, :, slice_idx]
        norm_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

        for vert_dict in vert_dicts:
            if slice_idx in vert_dict['slice_nos']:
                poly_idx = int(vert_dict['slice_nos'].index(slice_idx))
                poly = np.array(vert_dict['polys'][poly_idx], dtype=np.int32)

                # Рисуем полигон (желтый контур)
                cv2.polylines(color_img, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

                # Добавляем текст
                cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
                label = str(vert_dict['predicted_label'])
                cv2.putText(color_img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Преобразуем в PIL Image, если нужно
        images_with_masks.append(Image.fromarray(color_img))

    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan_volume)
    ivd_grades = spnt.grade_ivds(ivd_dicts)

    return images_with_masks, ivd_grades

if __name__ == "__main__":
    from dicom_io import load_dicoms_from_folder
    scan = load_dicoms_from_folder(r"F:\WorkSpace\Z-Union\test MRI", require_extensions=True)
    run_deepspine_pipeline(scan.volume, scan.pixel_spacing, scan.slice_thickness)
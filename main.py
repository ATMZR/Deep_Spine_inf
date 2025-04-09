import logging
import numpy as np
import os
from deepspine import DeepSpine

from PIL import Image
import cv2

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_two_rightmost_points(polygon, top_n=2):
    """
    Возвращает top_n точек с наибольшим значением по оси X.
    """
    # Сортируем точки по убыванию координаты x
    sorted_points = polygon[polygon[:, 0].argsort()[::-1]]
    return sorted_points[:top_n]

def find_lower_back_point(polygon):
    """
    Для нижнего позвонка:
    Из двух точек с максимальным x возвращает ту, у которой минимальное y.
    """
    two_rightmost = find_two_rightmost_points(polygon, top_n=2)
    lower_point = two_rightmost[np.argmin(two_rightmost[:, 1])]
    return lower_point

def find_upper_back_point(polygon):
    """
    Для верхнего позвонка:
    Из двух точек с максимальным x возвращает ту, у которой максимальное y.
    """
    two_rightmost = find_two_rightmost_points(polygon, top_n=2)
    upper_point = two_rightmost[np.argmax(two_rightmost[:, 1])]
    return upper_point

def get_label_lower_back_point(vert_dicts, label):
    """
    Получает точку задней стенки для нижнего позвонка (точку с максимальным x, затем выбирает по минимальному y)
    из vert_dicts для заданной метки.
    """
    for d in vert_dicts:
        if d['predicted_label'] == label:
            polygon = np.array(d['average_polygon'])
            return find_lower_back_point(polygon)
    return None

def get_label_upper_back_point(vert_dicts, label):
    """
    Получает точку задней стенки для верхнего позвонка (точку с максимальным x, затем выбирает по максимальному y)
    из vert_dicts для заданной метки.
    """
    for d in vert_dicts:
        if d['predicted_label'] == label:
            polygon = np.array(d['average_polygon'])
            return find_upper_back_point(polygon)
    return None

def compute_spondylolisthesis_displacement(vert_dicts, upper, lower):
    """
    Вычисляет листез (смещение) по задней стенке:
    для верхнего позвонка берётся точка с максимальным x и максимальным y,
    а для нижнего – точка с максимальным x и минимальным y.
    """
    upper_point = get_label_upper_back_point(vert_dicts, upper)
    lower_point = get_label_lower_back_point(vert_dicts, lower)

    if upper_point is None or lower_point is None:
        print(f"Не удалось найти точки задней стенки для {upper} или {lower}")
        return None

    # Вычисляем разницу по осям
    displacement_x = lower_point[0] - upper_point[0]

    displacement = displacement_x
    return displacement


def run_deepspine_pipeline(scan_volume, pixel_spacing, slice_thickness):
    spnt = DeepSpine(verbose=False)

    logger.info(f'Scan has {scan_volume.shape[-1]} sagittal slices, of dimension {scan_volume.shape[0]}x{scan_volume.shape[1]} ({pixel_spacing} mm pixel spacing) and {slice_thickness} mm slice thickness.')

    if len(scan_volume.shape) > 3:
        scan_volume = np.squeeze(scan_volume[:,:,[0], :])  # Убираем лишнюю ось

    vert_dicts = spnt.detect_vb(scan_volume, pixel_spacing)
    logger.info(f'{len(vert_dicts)} vertebrae detected: {[v["predicted_label"] for v in vert_dicts]}')

    images_with_masks = []

    save_dir = r'results/images'
    os.makedirs(save_dir, exist_ok=True)

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

        # Сохраняем изображение
        save_path = os.path.join(save_dir, f'slice_{slice_idx:03d}.png')
        cv2.imwrite(save_path, color_img)

        # Преобразуем в PIL Image, если нужно
        images_with_masks.append(Image.fromarray(color_img))

    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan_volume)
    ivd_grades = spnt.grade_ivds(ivd_dicts)
    ivd_grades['spondy_displacement'] = np.nan

    pairs = ivd_grades.index[ivd_grades['Spondylolisthesis'] == 1].tolist()
    if pairs:
        split = [p.split('-') for p in pairs]
        starts, ends = zip(*split)
        for upper, lower in zip(starts, ends):
            disp = compute_spondylolisthesis_displacement(vert_dicts, upper, lower)
            ivd_grades.at[f"{upper}-{lower}", 'spondy_displacement'] = round(disp, 4)
    ivd_grades.to_csv(r'results/ivd_grades.csv')



    return images_with_masks, ivd_grades

if __name__ == "__main__":
    from dicom_io import load_dicoms_from_folder
    scan = load_dicoms_from_folder(r"F:\WorkSpace\Z-Union\test MRI\T2_listez_2", require_extensions=True)
    run_deepspine_pipeline(scan.volume, scan.pixel_spacing, scan.slice_thickness)
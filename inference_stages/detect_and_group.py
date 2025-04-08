from shapely.geometry import Polygon, Point
import numpy as np
from .scan_preprocessing import split_into_patches_exhaustive
from .detection_post_processing import make_in_slice_detections


def detect_and_group(
    detection_net,
    scan,
    pixel_spacing=1,
    using_resnet=True,
    corner_threshold=0.5,
    centroid_threshold=0.5,
    group_across_slices_threshold=0.2,
    remove_single_slice_detections=True,
    debug=False,
):
    """
    Детектирует все углы позвонков, группирует их в отдельные позвонки и возвращает результаты.

    Параметры
    ---------
    detection_net : torch.nn.Module
        Модель VFR для детектирования ориентиров и углов позвонков.
    scan : np.ndarray
        3D-массив размера HxWxS, содержащий исследование.
    remove_excess_black_space : bool, по умолчанию True
        Удаление лишнего черного пространства по краям.
    pixel_spacing : float, по умолчанию 1
        Пространственное разрешение изображения в пикселях.
    plot_outputs : bool, по умолчанию False
        Отображать ли карты отклика для каналов детекции.
    using_resnet : bool, по умолчанию True
        Используется ли модель на основе ResNet. Влияет на нормализацию.
    corner_threshold : float от 0 до 1, по умолчанию 0.5
        Порог для каналов углов позвонков.
    centroid_threshold : float от 0 до 1, по умолчанию 0.5
        Порог для канала центра масс.
    group_across_slices_threshold : float, по умолчанию 0.2
        IOU-порог для объединения детекций между срезами.
    remove_single_slice_detections : bool, по умолчанию True
        Удалять ли объекты, детектированные только на одном срезе.
    device : str, по умолчанию "cuda:0"
        Устройство для инференса.
    debug : bool, по умолчанию False
        Если True — возвращает промежуточные данные.

    Возвращает
    ----------
    vert_dicts : list[dict]
        Список словарей, содержащих информацию о каждом позвонке:
        - 'average_polygon': усреднённый полигон
        - 'slice_nos': номера срезов, где найден объект
        - 'polys': полигоны на каждом срезе
    """

    # split the scan into different patches
    patches, transform_info_dicts = split_into_patches_exhaustive(
        scan, pixel_spacing=pixel_spacing, overlap_param=0.4, using_resnet=using_resnet
    )
    # group the detections made in each patch into slice level detections
    detection_dicts, patches_dicts = make_in_slice_detections(
        detection_net,
        patches,
        transform_info_dicts,
        scan.shape,
        corner_threshold,
        centroid_threshold,
    )

    vert_dicts = group_slice_detections(
        detection_dicts,
        iou_threshold=group_across_slices_threshold,
        remove_single_slice_detections=remove_single_slice_detections,
    )

    if not debug:
        return vert_dicts
    else:
        return vert_dicts, patches, patches_dicts, detection_dicts, transform_info_dicts


def group_slice_detections(
    detection_dicts, iou_threshold=0.1, remove_single_slice_detections=True
):
    """
    Группирует полигоны, детектированные на разных срезах, в 3D-объекты позвонков.

    Параметры
    ---------
    detection_dicts : list[dict]
        Список словарей, по одному на каждый срез. Каждый словарь описывает полигоны на этом срезе.
    iou_threshold : float, по умолчанию 0.1
        Порог IOU для объединения полигонов в один объект.
    remove_single_slice_detections : bool, по умолчанию True
        Удалять ли объекты, найденные только на одном срезе.

    Возвращает
    ----------
    vert_dicts : list[dict]
        Список словарей, описывающих каждый найденный позвонок.
    """
    vert_dicts = []
    # loop through slices
    for slice_no, slice_detections in enumerate(detection_dicts):
        # loop through vert bodies detected in each slice
        for polygon_in_slice in slice_detections["detection_polys"]:
            # now loop through previous detections and see if any match up
            overlaps_with_previous = False
            for vert in vert_dicts:
                try:
                    most_recent_polygon = vert["polys"][-1]
                    poly_ious = get_poly_iou(polygon_in_slice, most_recent_polygon)
                except:
                    poly_ious = 0
                if poly_ious > iou_threshold:
                    vert["polys"].append(polygon_in_slice)
                    vert["slice_nos"].append(slice_no)
                    # recalculate average poly for that vertebrae
                    vert["average_polygon"] = np.mean(vert["polys"], axis=0)
                    overlaps_with_previous = True
                    break
            # make new entry if doesn't overlap with any of the previous ones
            if overlaps_with_previous is False:
                vert_dicts.append(
                    {
                        "polys": [polygon_in_slice],
                        "average_polygon": polygon_in_slice,
                        "slice_nos": [slice_no],
                    }
                )
    remove_indices = []  # list to store indicies of verts to be deleted
    if remove_single_slice_detections:
        for vert_index, vert in enumerate(vert_dicts):
            if len(vert["polys"]) < 2:
                remove_indices.insert(
                    0, vert_index
                )  # insert at beginning to get reverse order
        [vert_dicts.pop(remove_index) for remove_index in remove_indices]

    # go through all vertebrae and check their average polygons dont overlap
    match_vert_idxs = []
    for vert_idx, vert_dict in enumerate(vert_dicts):
        for other_vert_idx, other_vert_dict in enumerate(vert_dicts):
            if vert_idx <= other_vert_idx:
                continue
            iou = get_poly_iou(
                vert_dict["average_polygon"], other_vert_dict["average_polygon"]
            )
            if iou > 0.05:
                match_vert_idxs.append([vert_idx, other_vert_idx])
                print('merging')

    # join the matching verts
    for matching_vert_pair in match_vert_idxs:
        vert_dicts[matching_vert_pair[0]]["polys"] += vert_dicts[matching_vert_pair[1]][
            "polys"
        ]
        vert_dicts[matching_vert_pair[0]]["slice_nos"] += vert_dicts[
            matching_vert_pair[1]
        ]["slice_nos"]
        vert_dicts[matching_vert_pair[0]]["average_polygon"] = np.mean(
            vert_dicts[matching_vert_pair[0]]["polys"], axis=0
        )
        print(f"merging vert {matching_vert_pair[0]} with {matching_vert_pair[1]}")

    # remove extra verts
    for matching_vert_pair in sorted(match_vert_idxs, key=lambda x: x[1], reverse=True):
        vert_dicts.pop(matching_vert_pair[1])

    # now sort list in height order
    vert_dicts.sort(key=lambda x: np.mean(np.array(x["average_polygon"])[:, 1]))
    return vert_dicts


def get_poly_iou(poly1, poly2):
    """
    Вычисляет IOU (пересечение над объединением) для двух полигонов.

    Параметры
    ---------
    poly1 : np.ndarray
        Массив 4x2 с вершинами первого полигона.
    poly2 : np.ndarray
        Массив 4x2 с вершинами второго полигона.

    Возвращает
    ----------
    iou : float
        Значение IOU между двумя полигонами.
    """

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    min_area = np.min([poly1.area, poly2.area])
    if union > 0:
        return intersection / union
    else:
        return 0


def red(x):
    y = np.zeros_like(x)
    return np.stack([x, y, y], axis=-1)


def blue(x):
    y = np.zeros_like(x)
    return np.stack([y, y, x], axis=-1)


def green(x):
    y = np.zeros_like(x)
    return np.stack([y, x, y], axis=-1)


def yellow(x):
    y = np.zeros_like(x)
    return np.stack([x, x, y], axis=-1)


def pink(x):
    y = np.zeros_like(x)
    return 0.5 * (yellow(x) + red(x))


def color(x):
    y = np.zeros_like(x)
    return np.stack([x, x, x], axis=-1)

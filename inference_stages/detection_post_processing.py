import numpy as np
from scipy.ndimage import zoom
import scipy.ndimage
from shapely.geometry import Polygon, Point


def make_in_slice_detections(
    detection_net,
    patches,
    transform_info_dicts,
    scan_shape,
    corner_threshold=0.3,
    centroid_threshold=0.5,
):
    # now resize patches
    detection_dicts = []
    patches_dicts = []

    for slice_idx in range(len(patches)):
        patches_array = np.stack(patches[slice_idx], axis=0)[:, None, :, :]
        net_output = detection_net.infer(patches_array)["output"]

        patches_dicts.append({"patches": patches_array, "net_output": net_output,
                              "landmark_points": {}, "landmark_arrows": {}})

        all_corners = {"points": {}, "arrows": {}}
        for corner_type in ["rt", "rb", "lb", "lt"]:
            all_corners["points"][corner_type] = []
            all_corners["arrows"][corner_type] = []
            patches_dicts[-1]["landmark_points"][corner_type] = []
            patches_dicts[-1]["landmark_arrows"][corner_type] = []


        scan_centroid_channel = np.zeros(scan_shape[0:2]).astype(float)
        centroid_channel_contributions = np.zeros_like(scan_centroid_channel)
        max_x = transform_info_dicts[slice_idx][0]["x2"]
        min_x = transform_info_dicts[slice_idx][0]["x1"]

        patch_edge_len = np.abs(max_x - min_x)


        for j in range(len(patches[slice_idx])):
            transform_info = transform_info_dicts[slice_idx][j]
            try:
                centroid_channel = net_output[:, 4, :, :]  # (B, H, W)

                target_h = transform_info["y2"] - transform_info["y1"]
                target_w = transform_info["x2"] - transform_info["x1"]

                zoomed = []
                for i in range(centroid_channel.shape[0]):
                    img = centroid_channel[i]  # (H, W)
                    zoom_factors = (
                        target_h / img.shape[0],
                        target_w / img.shape[1]
                    )
                    zoomed_img = zoom(img, zoom_factors, order=1)
                    zoomed.append(zoomed_img)

                resized_centroid_channels = np.stack(zoomed, axis=0)  # (B, H_new, W_new)
                resized_centroid_channels = np.expand_dims(resized_centroid_channels, axis=1)

                scan_centroid_channel[
                    transform_info["y1"] : transform_info["y2"],
                    transform_info["x1"] : transform_info["x2"],
                ] = (
                    scan_centroid_channel[
                        transform_info["y1"] : transform_info["y2"],
                        transform_info["x1"] : transform_info["x2"],
                    ]
                    + resized_centroid_channels[j, 0, :, :]
                )
                centroid_channel_contributions[
                    transform_info["y1"] : transform_info["y2"],
                    transform_info["x1"] : transform_info["x2"],
                ] += 1
            except Exception as E:
                print(
                    resized_centroid_channels.shape,
                    scan_centroid_channel[
                        transform_info["y1"] : transform_info["y2"],
                        transform_info["x1"] : transform_info["x2"],
                    ].shape,
                )
                print(str(E))

            for corner_idx, corner_type in enumerate(["rt", "rb", "lb", "lt"]):
                points = get_points(
                    net_output[j, corner_idx, :, :], threshold=corner_threshold
                )
                if len(points) == 0:
                    patches_dicts[-1]["landmark_points"][corner_type].append([])
                    patches_dicts[-1]["landmark_arrows"][corner_type].append([])
                    continue
                else:
                    if transform_info["y1"] < 0:
                        transform_info["y1"] = (
                            scan_centroid_channel.shape[0] + transform_info["y1"]
                        )
                    if transform_info["x1"] < 0:
                        transform_info["x1"] = (
                            scan_centroid_channel.shape[1] + transform_info["x1"]
                        )
                    transformed_patch_corners = (
                        points * patch_edge_len / 224
                        + np.array([transform_info["y1"], transform_info["x1"]])
                    )
                    all_corners["points"][corner_type].append(transformed_patch_corners)
                    arrows = np.zeros_like(points)
                    for idx, point in enumerate(points):
                        arrows[idx, 0] = net_output[
                            j, corner_idx + 9, point[0], point[1]
                        ]
                        arrows[idx, 1] = net_output[
                            j, corner_idx + 5, point[0], point[1]
                        ]
                    # scale arrows to original frame
                    all_corners["arrows"][corner_type].append(
                        arrows * patch_edge_len / 224
                    )
                    patches_dicts[-1]["landmark_points"][corner_type].append(points)
                    patches_dicts[-1]["landmark_arrows"][corner_type].append(arrows)


        disp_corners = {}
        for i in ["rt", "rb", "lb", "lt"]:
            if len(all_corners["points"][i]) > 0:
                all_corners["points"][i] = np.concatenate(
                    all_corners["points"][i], axis=0
                )
                all_corners["arrows"][i] = np.concatenate(
                    all_corners["arrows"][i], axis=0
                )
                disp_corners[i] = all_corners["points"][i] + all_corners["arrows"][i]
            else:
                disp_corners[i] = []

        centroid_channel_contributions[centroid_channel_contributions == 0] = 1
        scan_centroid_channel /= centroid_channel_contributions
        centroids = get_points(scan_centroid_channel, threshold=centroid_threshold)

        detection_polys = []
        arrows = all_corners["arrows"]
        corners = all_corners["points"]
        for centroid in centroids:
            if (
                len(disp_corners["lb"])
                and len(disp_corners["lt"])
                and len(disp_corners["rb"])
                and len(disp_corners["rt"])
            ):
                lt_dist = np.min(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                lt_arrow = arrows["lt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                ]
                closest_lt_corner = corners["lt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lt"], axis=1))
                ]
                lb_dist = np.min(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                lb_arrow = arrows["lb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                ]
                closest_lb_corner = corners["lb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["lb"], axis=1))
                ]
                rt_dist = np.min(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                rt_arrow = arrows["rt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                ]
                closest_rt_corner = corners["rt"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rt"], axis=1))
                ]
                rb_dist = np.min(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                rb_arrow = arrows["rb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                ]
                closest_rb_corner = corners["rb"][
                    np.argmin(np.linalg.norm(centroid - disp_corners["rb"], axis=1))
                ]
                indiv_arrows = [rt_arrow, rb_arrow, lb_arrow, lt_arrow]
                poly = Polygon(
                    [
                        closest_rt_corner,
                        closest_rb_corner,
                        closest_lb_corner,
                        closest_lt_corner,
                    ]
                )
                missing_arrows = arrows_threshold_check(
                    [
                        [rt_dist, rt_arrow],
                        [rb_dist, rb_arrow],
                        [lb_dist, lb_arrow],
                        [lt_dist, lt_arrow],
                    ]
                )
                detection_poly = np.array(
                    [
                        closest_rt_corner,
                        closest_rb_corner,
                        closest_lb_corner,
                        closest_lt_corner,
                    ]
                )
                if sum(missing_arrows) == 0:
                    if poly.is_valid and poly.contains(Point(centroid)):
                        detection_polys.append([[i[1], i[0]] for i in detection_poly])
                elif sum(missing_arrows) == 1:
                    for i, el in enumerate(missing_arrows):
                        if el:
                            detection_poly[i] = centroid + indiv_arrows[(i + 2) % 4]
                    detection_polys.append([[i[1], i[0]] for i in detection_poly])

        detection_polys = remove_polys_sharing_corners(detection_polys, all_corners)

        all_corners["centroid_heatmap"] = scan_centroid_channel
        all_corners["centroids"] = centroids
        all_corners["detection_polys"] = detection_polys
        all_corners["vert_index"] = [None] * len(detection_polys)
        detection_dicts.append(all_corners)

    return detection_dicts, patches_dicts


def remove_polys_sharing_corners(detection_polys, all_corners):

    # stores indicies of the polys to be removed
    polys_to_be_removed = []
    for detection_poly_idx, detection_poly in enumerate(detection_polys):
        for other_detection_poly_idx, other_detection_poly in enumerate(
            detection_polys
        ):
            # skip over overlaps with its self
            if detection_poly != other_detection_poly:
                # vectors between all the corners
                diff_arr = np.array(detection_poly) - np.array(other_detection_poly)
                if np.min(np.linalg.norm(diff_arr, axis=-1)) < 2:
                    # should go here if the detection polys have the same corner
                    detection_poly_internal_angles = get_internal_angles(detection_poly)
                    other_detection_poly_internal_angles = get_internal_angles(
                        other_detection_poly
                    )

                    if np.ptp(detection_poly_internal_angles) > np.ptp(
                        other_detection_poly_internal_angles
                    ):
                        polys_to_be_removed.append(detection_poly_idx)
                    else:
                        polys_to_be_removed.append(other_detection_poly_idx)

    polys_to_be_removed = sorted(list(set(polys_to_be_removed)), reverse=True)

    for poly_to_be_removed in polys_to_be_removed:
        detection_polys.pop(poly_to_be_removed)

    return detection_polys


def get_internal_angles(poly):

    poly_internal_angles = []
    poly = np.array(poly)
    for i in range(4):
        v1 = poly[i - 1, :] - poly[i, :]
        v2 = poly[(i + 1) % 4, :] - poly[i, :]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        poly_internal_angles.append(np.arccos(np.dot(v1, v2)))

    return poly_internal_angles


def arrows_threshold_check(arrows_arr):

    # arrows arr should be a list of 2-entry lists containing the distance form the centroid and the arrow in cartesian coords
    missing_arrows = [0, 0, 0, 0]  # binary mask to show missing corner
    for idx, el in enumerate(arrows_arr):
        if el[0] > 0.5 * np.linalg.norm(el[1]) or el[0] > 20:
            missing_arrows[idx] = 1
    return missing_arrows


def get_points(image, threshold=0.5):


    image = np.array(image)
    mask = image > threshold
    segmentation, no_labels = scipy.ndimage.label(mask)
    points = []
    for label in range(no_labels):
        point = np.unravel_index(
            np.argmax(image * (segmentation == (label + 1))), image.shape
        )
        points.append(point)

    return np.array(points)

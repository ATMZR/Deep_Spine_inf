import numpy as np
from typing import Union, List
from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient

class TritonModelWrapper:
    def __init__(self, model_name: str, url: str = "localhost:8000", model_version: str = "1"):
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpclient.InferenceServerClient(url=url)

    def infer(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        # Убедимся, что inputs — это словарь
        if not isinstance(inputs, dict):
            inputs = {"input": inputs}  # Заменить "input" на актуальное имя входа в модели

        triton_inputs = []
        for name, data in inputs.items():
            triton_input = httpclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
            triton_input.set_data_from_numpy(data)
            triton_inputs.append(triton_input)

        # Выходы
        if self.model_name != "grading":
            triton_outputs = [httpclient.InferRequestedOutput("output")]
            output_names = ["output"]
        else:
            output_names = [
                "output_pf", "output_nar", "output_ccs", "output_spn", "output_ued",
                "output_led", "output_umc", "output_lmc", "output_fsl", "output_fsr", "output_hrn"
            ]
            triton_outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

        # Инференс
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )

        # Собираем словарь всех выходов
        outputs = {name: response.as_numpy(name) for name in output_names}
        return outputs


class DeepSpine:
    def __init__(self, triton_url: str = "localhost:8000", scan_type: str = "lumbar", verbose: bool = True) -> None:
        assert scan_type in ["lumbar", "whole"]
        self.triton_url = triton_url
        self.remove_black_space = scan_type != "lumbar"
        self.corner_threshold = 0.6
        self.centroid_threshold = 0.6
        self.group_across_slices_threshold = 0.2
        self.verbose = verbose

        if verbose:
            print("Connecting to Triton and setting up models...")

        self.detection_model = TritonModelWrapper("detect-vfr", url=triton_url)
        self.appearance_model = TritonModelWrapper("appearance", url=triton_url)
        self.context_model = TritonModelWrapper("context", url=triton_url)
        self.grading_model = TritonModelWrapper("grading", url=triton_url)

    def detect_vb(self, volume: np.ndarray, pixel_spacing: Union[np.ndarray, List[float]], **kwargs):
        from inference_stages import detect_and_group, extract_volumes, label_verts  # адаптируй под себя

        detect_ans = detect_and_group(
            detection_net=self.detection_model,
            scan=volume,
            using_resnet=True,
            corner_threshold=self.corner_threshold,
            centroid_threshold=self.centroid_threshold,
            group_across_slices_threshold=self.group_across_slices_threshold,
            pixel_spacing=pixel_spacing,
            **kwargs
        )
        vert_dicts = extract_volumes(volume, detect_ans)
        vert_dicts = label_verts(
            vert_dicts=vert_dicts,
            scan=volume,
            pixel_spacing=pixel_spacing,
            appearance_net=self.appearance_model,
            context_net=self.context_model,
            **kwargs
        )
        return vert_dicts

    def get_ivds_from_vert_dicts(self, vert_dicts, scan_volume):
        from inference_stages import vert_dicts_to_classification_format, get_all_ivd_vol, get_ivd_level_names

        (
            all_vb_x,
            all_vb_y,
            all_vb_mid,
            all_vb_label,
            vb_level_names,
        ) = vert_dicts_to_classification_format(vert_dicts, scan_volume.shape[-1])
        ivds = get_all_ivd_vol(scan_volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label)
        ivd_level_names = get_ivd_level_names(vb_level_names)
        return [{"volume": ivd, "level_name": name} for ivd, name in zip(ivds, ivd_level_names)]

    def grade_ivds(self, ivd_dicts):
        from inference_stages import classify_ivd_v2_resnet, format_gradings

        volumes = [d["volume"] for d in ivd_dicts]
        levels = [d["level_name"] for d in ivd_dicts]
        gradings = classify_ivd_v2_resnet(self.grading_model, volumes)
        return format_gradings(gradings, levels)

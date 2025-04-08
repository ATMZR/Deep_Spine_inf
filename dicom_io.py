import os
from typing import List, Union
import glob
import numpy as np
from pydicom import dcmread, FileDataset, DataElement
from pydicom.tag import Tag


class SpinalScan:
    def __init__(
        self,
        volume: np.array,
        pixel_spacing: Union[np.array, list],
        slice_thickness: Union[float, int],
    ) -> None:
        """
        Инициализация объекта скана позвоночника, который используется в DeepSpine.

        Параметры
        ----------
        volume : np.array
            3D-массив данных скана (обычно размер: высота x ширина x количество срезов)
        pixel_spacing : Union[np.array, list]
            Расстояние между пикселями в срезах (в мм). Обычно порядок: высота, ширина.
        slice_thickness : Union[float, int]
            Расстояние между соседними срезами (в мм).
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness



def load_dicoms(
        paths: List[Union[os.PathLike, str, bytes]],
        require_extensions: bool = True,
        metadata_overwrites: dict = {},
    ) -> SpinalScan:
        '''
        Создает объект SpinalScan из DICOM-файлов.
        Производит проверки наличия ключевых тегов и убеждается, что срезы являются сагиттальными.

        Параметры
        ----------
        paths : List[Union[os.PathLike, str, bytes]]
            Список путей к DICOM-файлам.
        require_extensions : bool
            Если True, проверяется, что у всех путей расширение ".dcm".
        metadata_overwrites : dict
            Словарь для перезаписи метаданных в скане (PixelSpacing, SliceThickness, ImageOrientationPatient).

        Возвращает
        -------
        SpinalScan
            Объект, представляющий скан, сформированный из DICOM-файлов.
        '''
        # Проверка расширений файлов
        if require_extensions:
            assert all(
                [".dcm" == path[-4:] for path in paths]
            ), "Все пути должны иметь расширение .dcm. Если хотите игнорировать расширение, установите require_extensions=False"

        # Читаем все DICOM-файлы
        dicom_files = [dcmread(path) for path in paths]

        # Перезаписываем теги согласно metadata_overwrites для каждого файла
        for idx, dicom_file in enumerate(dicom_files):
            dicom_files[idx] = overwrite_tags(dicom_file, metadata_overwrites)

        # Проверяем наличие нужных тегов и что срезы сагиттальные
        for dicom_idx, dicom_file in enumerate(dicom_files):
            missing_tags = check_missing_tags(dicom_file)
            if len(missing_tags) > 0:
                raise ValueError(
                    f"В файле {paths[dicom_idx]} отсутствуют теги: {missing_tags}"
                )
            is_sagittal = is_sagittal_dicom_slice(dicom_file)
            if not is_sagittal:
                raise ValueError(
                    f"Файл {paths[dicom_idx]} не является сагиттальной DICOM-срезом"
                )
        # Сортировка срезов по порядковому номеру (согласно тегу InstanceNumber)
        dicom_files = sorted(
            dicom_files, key=lambda dicom_file: dicom_file.InstanceNumber
        )

        # Вычисляем среднее значение PixelSpacing и толщины среза
        pixel_spacing = np.mean(
            [np.array(dicom_file.PixelSpacing) for dicom_file in dicom_files]
        )
        slice_thickness = np.mean(
            [np.array(dicom_file.SliceThickness) for dicom_file in dicom_files]
        )
        # Собираем объем скана, складывая массивы пиксельных данных по новой оси
        volume = np.stack(
            [np.array(dicom_file.pixel_array) for dicom_file in dicom_files], axis=-1
        )

        return SpinalScan(
            volume=volume, pixel_spacing=pixel_spacing, slice_thickness=slice_thickness
        )


def is_sagittal_dicom_slice(dicom_file: FileDataset) -> bool:
    '''
    Проверяет, является ли DICOM-срез сагиттальным.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл для проверки.

    Возвращает
    -------
    bool
        True, если срез сагиттальный; иначе False.
    '''
    if Tag("ImageOrientationPatient") in dicom_file:
        image_orientation = np.array(dicom_file.ImageOrientationPatient).round()
        # Если первые и четвертые значения (обычно соответствующие первому направлению) равны нулю,
        # то срез сагиттальный.
        if (image_orientation[[0, 3]] == [0, 0]).all():
            return True
        else:
            return False
    else:
        raise ValueError("Метаданные ImageOrientationPatient отсутствуют в DICOM-файле")


def overwrite_tags(dicom_file: FileDataset, metadata_overwrites: dict) -> FileDataset:
    '''
    Перезаписывает теги в DICOM-файле. В данный момент поддерживается перезапись:
    PixelSpacing, SliceThickness и ImageOrientationPatient.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл, в котором нужно изменить значения.
    metadata_overwrites : dict
        Словарь метаданных для перезаписи (например, PixelSpacing, SliceThickness, ImageOrientationPatient).

    Возвращает
    -------
    FileDataset
        DICOM-файл с изменёнными значениями.
    '''
    possible_overwrites = {
        "PixelSpacing": "DS",
        "SliceThickness": "DS",
        "ImageOrientationPatient": "DS",
    }

    for tag, value in metadata_overwrites.items():
        if tag not in possible_overwrites:
            raise NotImplementedError(f"Перезапись тега {tag} не поддерживается")
        else:
            if Tag(tag) in dicom_file:
                dicom_file[Tag(tag)] = DataElement(
                    Tag(tag), possible_overwrites[tag], value
                )
            else:
                dicom_file.add_new(Tag(tag), possible_overwrites[tag], value)
    return dicom_file


def check_missing_tags(dicom_file: FileDataset) -> List[str]:
    '''
    Определяет, какие теги отсутствуют в DICOM-файле.
    Требуются теги: PixelData, PixelSpacing, SliceThickness, InstanceNumber.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл для проверки.

    Возвращает
    -------
    List[str]
        Список отсутствующих тегов.
    '''
    required_tags = ["PixelData", "PixelSpacing", "SliceThickness", "InstanceNumber"]
    missing_tags = [
        tag_name for tag_name in required_tags if Tag(tag_name) not in dicom_file
    ]
    return missing_tags


def is_dicom_file(path: Union[os.PathLike, str, bytes]) -> bool:
    '''
    Проверяет, является ли файл DICOM-файлом.

    Параметры
    ----------
    path : Union[os.PathLike, str, bytes]
        Путь к файлу.

    Возвращает
    -------
    bool
        True, если файл DICOM, иначе False.
    '''
    try:
        dcmread(path)
        return True
    except:
        return False


def load_dicoms_from_folder(
    path: Union[os.PathLike, str, bytes],
    require_extensions: bool = True,
    metadata_overwrites: dict = {},
) -> SpinalScan:
    '''
    Загружает DICOM-скан из папки, содержащей срезы.

    Параметры
    ----------
    path : Union[os.PathLike, str, bytes]
        Путь к папке с DICOM-срезами.
    require_extensions : bool
        Если True, проверяет, что все DICOM-файлы имеют расширение .dcm.
    metadata_overwrites : dict
        Словарь метаданных для перезаписи.

    Возвращает
    -------
    SpinalScan
        Объект, представляющий скан позвоночника, сформированный из DICOM-файлов.
    '''
    slices = [f for f in glob.glob(os.path.join(path, "*")) if is_dicom_file(f)]
    return load_dicoms(slices, require_extensions, metadata_overwrites)

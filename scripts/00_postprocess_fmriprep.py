#!/usr/bin/env python3
"""
Постобработка производных fMRIPrep: парцелляция, подавление шума, цензурирование,
построение матриц функциональной связности.

Выборка: ABIDE NYU, 184 испытуемых, fMRIPrep 25.2.5 LTS.
Основная парцелляция: AAL3, объёмная, мозжечок и червь отключены.

Что делает
----------
Для каждого испытуемого и каждой заданной парцелляции:
  1. отбрасывает DUMMY начальных объёмов;
  2. извлекает средние временные ряды по областям с контролем покрытия
     каждой области полем зрения испытуемого;
  3. строит матрицу нуисансных регрессоров и удаляет их проекцию;
  4. при необходимости фильтрует нижних частот;
  5. отбрасывает цензурированные объёмы;
  6. считает матрицу связности с преобразованием Фишера.

Варианты подавления шума считаются все четыре; основной объявлен в CONFIG и
записывается в манифест. Прочие три сохраняются для анализа устойчивости.

Порядок операций
----------------
Парцелляция выполняется ДО регрессии. Это допустимо: усреднение по области,
регрессия и фильтрация суть линейные операции, причём первая действует по
пространству, а две другие по времени, поэтому они коммутируют. Выигрыш —
работа с матрицами T x R вместо T x V.

Цензурированные объёмы входят в план как индикаторные регрессоры (по одному на
объём). Это в точности эквивалентно исключению их из подгонки регрессии, но
сохраняет непрерывность ряда для последующей фильтрации. Перед фильтрацией
значения в цензурированных объёмах заменяются линейной интерполяцией, после
фильтрации эти объёмы отбрасываются. Порядок «интерполяция, фильтрация,
отбрасывание» выбран потому, что фильтрация ряда с выколотыми точками
разносит артефакт движения по всей записи.

Регрессоры косинусного базиса (cosine*) включаются обязательно. Компоненты
aCompCor вычисляются fMRIPrep на данных после высокочастотного фильтра с
периодом отсечения 128 с, и без соответствующих косинусных регрессоров
подавление шума некорректно. Они же выполняют роль детренда, поэтому явные
линейный и квадратичный тренды из прежней редакции плана избыточны.

Покрытие областей
-----------------
AAL3 содержит мелкие подкорковые и стволовые ядра, а поле зрения ABIDE NYU
(33 среза по 4 мм) обрезает нижние отделы. Для каждой области считается доля
её вокселей, попавших в маску мозга испытуемого. Область с покрытием ниже
COVERAGE_MIN помечается непригодной у данного испытуемого: её ряд заполняется
NaN, а сама область попадает в отчёт. Решение о том, отбрасывать ли такие
области по всей выборке или отбрасывать испытуемых, принимается на этапе
анализа по файлам coverage/ и coverage_summary.tsv. Это прямое следствие
пловдивского опыта, где тринадцать областей оказались константными у 69
испытуемых именно из-за поля зрения, и обнаружилось это поздно.

Запуск
------
    python3 postproc_abide.py --derivatives /home/nctlab/mri-work/derivatives \
                              --out         /home/nctlab/mri-work/postproc \
                              --atlas-dir   /home/nctlab/atlases \
                              --n-jobs 6

    python3 postproc_abide.py ... --subjects sub-0050952 sub-0050953

Экспорт для ветви зигзага
-------------------------
    python3 postproc_abide.py --out /home/nctlab/mri-work/postproc \
                              --export-window-scan \
                              --phenotype Phenotypic_V1_0b_preprocessed1.csv \
                              --centroids centroids_aal3.tsv \
                              --expect-nodes 134

Собирает window_scan_input.npz: ряды ДО отбрасывания цензурированных объёмов,
маску цензурирования, общий набор узлов по правилу полной выборки и ковариаты
в том же порядке. Требует предварительного полного прогона постобработки при
SAVE_UNCENSORED=True.

Требуемые файлы в --atlas-dir
-----------------------------
    AAL3v1.nii.gz        объёмный атлас
    AAL3v1.nii.txt       таблица меток: индекс, имя, значение в файле

Замечание о пространстве. AAL3 распространяется в пространстве SPM, которое
ближе к MNI152NLin6Asym, тогда как производные fMRIPrep здесь в
MNI152NLin2009cAsym. Атлас приводится к сетке функциональных данных ближайшим
соседом; расхождение пространств порядка нескольких миллиметров. Для крупных
корковых областей это несущественно, для мелких стволовых ядер существенно, но
они и так отсеиваются по покрытию. Если потребуется точность, у fMRIPrep можно
дозапросить выход в MNI152NLin6Asym: полного пересчёта это не требует,
достаточно повторного прогона с другим --output-spaces.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nb
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------- #
# CONFIG — всё, что фиксируется в протоколе до расчёта
# --------------------------------------------------------------------------- #

CONFIG = dict(
    TR=2.0,
    DUMMY=10,
    TASK="rest",
    SPACE_VOL="MNI152NLin2009cAsym",
    RES_VOL="2",
    SPACE_SURF="fsaverage5",

    # Парцелляции. Первая по списку — основная.
    PARCELLATIONS=(
        dict(name="aal3", type="volume",
             nii="AAL3v1.nii.gz", labels="AAL3v1.nii.txt",
             drop_pattern=r"^(Cerebelum|Cerebellum|Vermis)"),
        # Поверхностная ветвь оставлена в коде и отключена. Включать, если
        # понадобится мультимасштабная конструкция: вложенных разрешений у
        # AAL3 нет, у Schaefer есть.
        # dict(name="schaefer200_tian", type="surface", n_parcels=200,
        #      subcortex_nii="Tian_Subcortex_S2_3T_2009cAsym.nii.gz",
        #      subcortex_labels="Tian_Subcortex_S2_3T_label.txt"),
    ),
    MAIN_PARCELLATION="aal3",
    COVERAGE_MIN=0.50,            # минимальная доля вокселей области в маске мозга

    N_ACOMPCOR=5,                 # компоненты маски combined
    MOTION_24=True,
    USE_COSINE=True,

    # Цензурирование: колонки motion_outlier_* fMRIPrep,
    # то есть FD > 0.5 мм или std_DVARS > 1.5
    CENSOR_SOURCE="fmriprep_motion_outlier",

    VARIANTS=("noGSR_bp", "noGSR_nobp", "GSR_bp", "GSR_nobp"),
    MAIN_VARIANT="noGSR_bp",
    LOWPASS_HZ=0.10,              # нижняя граница задана косинусами (~0.008 Гц)
    FILTER_ORDER=2,

    # Экспорт для ветви зигзага: ряды сохраняются ДО отбрасывания цензурированных
    # объёмов, вместе с маской. Скользящее окно по ряду с выброшенными точками
    # проезжает разрыв молча: окно из W подряд идущих ОСТАВШИХСЯ отсчётов
    # охватывает разный реальный интервал у разных испытуемых.
    SAVE_UNCENSORED=True,

    # Номинальный период отсечения высокочастотного фильтра fMRIPrep, задающий
    # нижнюю границу полосы. Значение по умолчанию; проверяется по числу
    # фактических регрессоров cosine* (см. cosine_cutoff_interval).
    COSINE_CUTOFF_S_NOMINAL=128.0,

    # Отбраковка применяется на этапе анализа, не здесь; приводится для протокола
    EXCLUSION_PRIMARY="median FD <= 0.14 mm and n_keep >= 120",
    EXCLUSION_SENSITIVITY=("mean FD <= 0.20 mm", "mean FD over retained <= 0.16 mm"),
)

log = logging.getLogger("postproc")


# --------------------------------------------------------------------------- #
# Атласы
# --------------------------------------------------------------------------- #

def read_label_table(path: Path) -> pd.DataFrame:
    """Таблица меток вида «индекс имя [значение]».

    В AAL3v1 индексы 35, 36, 81, 82 отсутствуют: прежние Cingulum_Ant и
    Thalamus разбиты на подобласти, вынесенные в конец списка. Нумерация
    доходит до 170 при 166 реальных областях, поэтому опираться на порядковый
    номер строки нельзя, только на значение метки в файле изображения.
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2 or not re.fullmatch(r"\d+", parts[0]):
            continue
        rows.append((int(parts[0]), parts[1]))
    if not rows:
        raise RuntimeError(f"Не удалось разобрать метки из {path}")
    return pd.DataFrame(rows, columns=["value", "label"])


def load_volume_atlas(atlas_dir: Path, spec: dict) -> tuple:
    """Возвращает (nifti атласа, таблица меток после отбрасывания по шаблону)."""
    nii, txt = atlas_dir / spec["nii"], atlas_dir / spec["labels"]
    for p in (nii, txt):
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл атласа: {p}")
    lab = read_label_table(txt)
    pat = spec.get("drop_pattern")
    if pat:
        keep = ~lab.label.str.match(pat)
        lab = lab[keep].reset_index(drop=True)
    return nb.load(str(nii)), lab


def parcellate_volume(nii_path: Path, mask_path: Path | None,
                      atlas_img, labels: pd.DataFrame,
                      coverage_min: float) -> tuple:
    """Средние ряды по областям и покрытие каждой области маской мозга.

    Возвращает (Y: T x R, coverage: R, n_vox_atlas: R).
    Области с покрытием ниже порога заполняются NaN.
    """
    from nilearn.image import resample_to_img

    img = nb.load(str(nii_path))
    ref = nb.Nifti1Image(np.zeros(img.shape[:3], np.int8), img.affine)
    atl = atlas_img
    if atl.shape != img.shape[:3] or not np.allclose(atl.affine, img.affine):
        atl = resample_to_img(atl, ref, interpolation="nearest",
                              force_resample=True, copy_header=True)
    a = np.asarray(atl.dataobj).astype(np.int32)

    brain = (np.asarray(nb.load(str(mask_path)).dataobj) > 0
             if mask_path is not None and Path(mask_path).exists()
             else np.ones(a.shape, bool))

    d = np.asarray(img.dataobj)                              # X Y Z T
    R = len(labels)
    Y = np.full((d.shape[3], R), np.nan)
    cov = np.zeros(R)
    nvox = np.zeros(R, int)
    for j, v in enumerate(labels.value.to_numpy()):
        m = a == v
        nvox[j] = int(m.sum())
        if nvox[j] == 0:
            continue
        mb = m & brain
        cov[j] = mb.sum() / nvox[j]
        if cov[j] >= coverage_min and mb.sum() > 0:
            Y[:, j] = d[mb].mean(axis=0)
    return Y, cov, nvox


def parcellate_surface(gii_path: Path, annot_path: Path) -> np.ndarray:
    """Средние ряды по областям одного полушария (поверхностная ветвь). T x R."""
    g = nb.load(str(gii_path))
    data = np.column_stack([d.data for d in g.darrays])       # V x T
    lab, _, names = nb.freesurfer.read_annot(str(annot_path))
    n_roi = len(names) - 1                                    # метка 0 — медиальная стенка
    out = np.full((data.shape[1], n_roi), np.nan)
    for i in range(1, n_roi + 1):
        m = lab == i
        if m.sum():
            out[:, i - 1] = data[m].mean(axis=0)
    return out


def build_parcellation(spec: dict, atlas_dir: Path, src: dict,
                       dummy: int) -> tuple:
    """Временные ряды, метки, покрытие и объём областей для одной парцелляции."""
    if spec["type"] == "volume":
        atl, lab = load_volume_atlas(atlas_dir, spec)
        Y, cov, nvox = parcellate_volume(src["vol"], src["mask"], atl, lab,
                                         CONFIG["COVERAGE_MIN"])
        return Y[dummy:], lab.label.tolist(), cov, nvox

    if spec["type"] == "surface":
        fs = atlas_dir / "fsaverage5"
        stem = f"Schaefer2018_{spec['n_parcels']}Parcels_7Networks_order.annot"
        lh = parcellate_surface(src["surf_L"], fs / f"lh.{stem}")[dummy:]
        rh = parcellate_surface(src["surf_R"], fs / f"rh.{stem}")[dummy:]
        _, _, names_l = nb.freesurfer.read_annot(str(fs / f"lh.{stem}"))
        _, _, names_r = nb.freesurfer.read_annot(str(fs / f"rh.{stem}"))
        ctx = ([f"LH_{n.decode()}" for n in names_l[1:]] +
               [f"RH_{n.decode()}" for n in names_r[1:]])

        names = [l.strip() for l in
                 (atlas_dir / spec["subcortex_labels"]).read_text().splitlines()
                 if l.strip()]
        sub_lab = pd.DataFrame({"value": np.arange(1, len(names) + 1),
                                "label": names})
        sc, cov_s, nv_s = parcellate_volume(
            src["vol"], src["mask"], nb.load(str(atlas_dir / spec["subcortex_nii"])),
            sub_lab, CONFIG["COVERAGE_MIN"])
        Y = np.column_stack([lh, rh, sc[dummy:]])
        cov = np.concatenate([np.ones(len(ctx)), cov_s])
        nvox = np.concatenate([np.full(len(ctx), -1), nv_s])
        return Y, ctx + names, cov, nvox

    raise ValueError(f"Неизвестный тип парцелляции: {spec['type']}")


# --------------------------------------------------------------------------- #
# Нуисанс
# --------------------------------------------------------------------------- #

MOTION_BASE = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
MOTION_24 = [f"{b}{s}" for b in MOTION_BASE
             for s in ("", "_derivative1", "_power2", "_derivative1_power2")]


def select_acompcor(conf_json: Path, n: int) -> list:
    """Первые n компонент aCompCor маски combined в порядке объяснённой дисперсии.

    Отбор по маске обязателен: в таблице присутствуют компоненты трёх масок
    (combined, CSF, WM), и брать первые по имени значит смешивать их.
    """
    if not conf_json.exists():
        log.warning("Нет %s, беру a_comp_cor_00..%02d без проверки маски",
                    conf_json.name, n - 1)
        return [f"a_comp_cor_{i:02d}" for i in range(n)]
    meta = json.loads(conf_json.read_text())
    comps = [(k, v) for k, v in meta.items()
             if k.startswith("a_comp_cor") and v.get("Mask") == "combined"]
    comps.sort(key=lambda kv: kv[1].get("CumulativeVarianceExplained", 1.0))
    if len(comps) < n:
        raise RuntimeError(f"Компонент combined всего {len(comps)}, требуется {n}")
    return [k for k, _ in comps[:n]]


def build_design(conf: pd.DataFrame, conf_json: Path, gsr: bool) -> tuple:
    """Матрица нуисанса и булев вектор цензурируемых объёмов."""
    cols = list(MOTION_24) if CONFIG["MOTION_24"] else list(MOTION_BASE)
    cols += select_acompcor(conf_json, CONFIG["N_ACOMPCOR"])
    if CONFIG["USE_COSINE"]:
        cols += sorted(c for c in conf.columns if c.startswith("cosine"))
    if gsr:
        cols += ["global_signal", "global_signal_derivative1",
                 "global_signal_power2", "global_signal_derivative1_power2"]

    missing = [c for c in cols if c not in conf.columns]
    if missing:
        raise RuntimeError(f"Отсутствуют колонки регрессоров: {missing}")

    X = np.nan_to_num(conf[cols].to_numpy(dtype=float), nan=0.0)
    X = np.column_stack([np.ones(len(X)), X])

    out_cols = [c for c in conf.columns if c.startswith("motion_outlier")]
    censor = (conf[out_cols].to_numpy().sum(axis=1) > 0) if out_cols \
        else np.zeros(len(conf), bool)
    if censor.any():
        spikes = np.zeros((len(conf), int(censor.sum())))
        spikes[np.where(censor)[0], np.arange(censor.sum())] = 1.0
        X = np.column_stack([X, spikes])
    return X, censor


def cosine_cutoff_interval(n_cosine: int, n_vol: int, tr: float) -> tuple:
    """Интервал периода отсечения, совместимый с числом регрессоров cosine*.

    fMRIPrep строит косинусный базис как DCT с числом функций
    ``n = floor(2 * D / P)``, где ``D = n_vol * tr`` — длительность записи,
    ``P`` — период отсечения. Обращение даёт интервал

        2D / (n + 1)  <  P  <=  2D / n

    Проверка грубая: одна лишняя или недостающая функция сдвигает границы
    заметно. Она и не претендует на точность — её задача поймать изменение
    настройки fMRIPrep, при котором нижняя граница полосы уедет, а всё
    остальное продолжит считаться как ни в чём не бывало.

    Возвращает ``(P_low, P_high)`` в секундах либо ``(nan, nan)`` при n = 0.
    """
    if n_cosine <= 0:
        return (float("nan"), float("nan"))
    D = n_vol * tr
    return (2.0 * D / (n_cosine + 1), 2.0 * D / n_cosine)


def regress_out(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Остатки Y после проекции на пространство столбцов X. Y: T x R."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def interpolate_censored(Y: np.ndarray, censor: np.ndarray) -> np.ndarray:
    """Линейная интерполяция значений в цензурированных объёмах."""
    if not censor.any():
        return Y
    Z = Y.copy()
    t = np.arange(len(Y))
    good = ~censor
    if good.sum() < 2:
        raise RuntimeError("Слишком мало нецензурированных объёмов для интерполяции")
    for j in range(Y.shape[1]):
        Z[censor, j] = np.interp(t[censor], t[good], Y[good, j])
    return Z


def lowpass(Y: np.ndarray, tr: float, hz: float, order: int) -> np.ndarray:
    nyq = 0.5 / tr
    if hz >= nyq:
        log.warning("Частота среза %.3f не ниже найквистовой %.3f, фильтр не применён",
                    hz, nyq)
        return Y
    b, a = butter(order, hz / nyq, btype="low")
    pad = min(3 * max(len(a), len(b)), len(Y) - 1)
    return filtfilt(b, a, Y, axis=0, padlen=pad)


# --------------------------------------------------------------------------- #
# Обработка одного испытуемого
# --------------------------------------------------------------------------- #

def find_inputs(deriv: Path, sub: str, need_surface: bool) -> dict:
    f = deriv / sub / "func"
    task = CONFIG["TASK"]
    conf = f / f"{sub}_task-{task}_desc-confounds_timeseries.tsv"
    if not conf.exists():
        raise FileNotFoundError(f"Нет таблицы регрессоров для {sub}")
    got = dict(conf=conf, conf_json=conf.with_suffix(".json"))

    vol = sorted(f.glob(f"{sub}_task-{task}*space-{CONFIG['SPACE_VOL']}"
                        f"*res-{CONFIG['RES_VOL']}*desc-preproc_bold.nii.gz"))
    if not vol:
        vol = sorted(f.glob(f"{sub}_task-{task}*space-{CONFIG['SPACE_VOL']}"
                            f"*desc-preproc_bold.nii.gz"))
    if not vol:
        raise FileNotFoundError(f"Нет объёмного выхода для {sub}")
    got["vol"] = vol[0]
    mask = str(vol[0]).replace("desc-preproc_bold", "desc-brain_mask")
    got["mask"] = Path(mask) if Path(mask).exists() else None
    if got["mask"] is None:
        log.warning("%s: нет маски мозга, покрытие областей не проверяется", sub)

    if need_surface:
        for hemi in ("L", "R"):
            cand = sorted(f.glob(f"{sub}_task-{task}*hemi-{hemi}*"
                                 f"{CONFIG['SPACE_SURF']}*.func.gii"))
            if not cand:
                raise FileNotFoundError(f"Нет поверхностного выхода {hemi} для {sub}")
            got[f"surf_{hemi}"] = cand[0]
    return got


def process_subject(sub: str, deriv: Path, out: Path, atlas_dir: Path) -> dict:
    log.info("%s: старт", sub)
    specs = [dict(s) for s in CONFIG["PARCELLATIONS"]]
    src = find_inputs(deriv, sub, any(s["type"] == "surface" for s in specs))

    conf = pd.read_csv(src["conf"], sep="\t")
    dummy = CONFIG["DUMMY"]
    conf = conf.iloc[dummy:].reset_index(drop=True)

    parc = {}
    for spec in specs:
        parc[spec["name"]] = build_parcellation(spec, atlas_dir, src, dummy)

    T = next(iter(parc.values()))[0].shape[0]
    if len(conf) != T:
        raise RuntimeError(f"{sub}: {len(conf)} строк регрессоров против {T} объёмов")

    rec = dict(subject=sub, n_volumes_raw=T + dummy, n_volumes=T)

    # отчёт по покрытию — один раз, вне цикла вариантов
    cov_rows = []
    for name, (Y, labels, cov, nvox) in parc.items():
        bad = np.isnan(Y).any(axis=0)
        rec[f"n_roi_{name}"] = int(len(labels))
        rec[f"n_dropped_roi_{name}"] = int(bad.sum())
        rec[f"min_coverage_{name}"] = float(np.min(cov)) if len(cov) else np.nan
        cov_rows += [dict(subject=sub, parcellation=name, label=l,
                          n_vox_atlas=int(v), coverage=float(c), dropped=bool(b))
                     for l, v, c, b in zip(labels, nvox, cov, bad)]
    (out / "coverage").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov_rows).to_csv(out / "coverage" / f"{sub}_coverage.tsv",
                                  sep="\t", index=False)

    for variant in CONFIG["VARIANTS"]:
        gsr = variant.startswith("GSR")
        bp = variant.endswith("_bp")
        X, censor = build_design(conf, src["conf_json"], gsr=gsr)
        n_keep = int((~censor).sum())
        rec.update(n_censored=int(censor.sum()), n_keep=n_keep)
        if n_keep < 60:
            log.warning("%s: остаётся %d объёмов, обработка продолжена, "
                        "решение об исключении на этапе анализа", sub, n_keep)

        for name, (Y, labels, cov, nvox) in parc.items():
            bad = np.isnan(Y).any(axis=0)
            d = out / variant / name
            (d / "timeseries").mkdir(parents=True, exist_ok=True)
            (d / "connectivity").mkdir(parents=True, exist_ok=True)

            Res = regress_out(np.nan_to_num(Y, nan=0.0), X)
            if bp:
                Res = interpolate_censored(Res, censor)
                Res = lowpass(Res, CONFIG["TR"], CONFIG["LOWPASS_HZ"],
                              CONFIG["FILTER_ORDER"])

            # Полный ряд до отбрасывания цензурированных объёмов. В вариантах
            # _bp значения в цензурированных объёмах интерполированы, в _nobp —
            # это остатки регрессии как есть. Маска сохраняется рядом, чтобы
            # потребитель решал сам, а не угадывал.
            if CONFIG["SAVE_UNCENSORED"]:
                Full = Res.copy()
                Full[:, bad] = np.nan
                np.save(d / "timeseries" / f"{sub}_ts_full.npy",
                        Full.astype(np.float32))
                np.save(d / "timeseries" / f"{sub}_censor.npy", censor)

            Res = Res[~censor]
            Res[:, bad] = np.nan
            np.save(d / "timeseries" / f"{sub}_ts.npy", Res.astype(np.float32))

            with np.errstate(invalid="ignore"):
                C = np.corrcoef(Res, rowvar=False)
            C = np.clip(C, -0.999999, 0.999999)
            Z = np.arctanh(C)
            np.fill_diagonal(Z, 0.0)
            np.save(d / "connectivity" / f"{sub}_fisherz.npy", Z.astype(np.float32))

            if not (d / "labels.tsv").exists():
                pd.DataFrame({"index": np.arange(1, len(labels) + 1),
                              "label": labels}).to_csv(d / "labels.tsv",
                                                       sep="\t", index=False)

    n_cosine = int(sum(c.startswith("cosine") for c in conf.columns))
    p_low, p_high = cosine_cutoff_interval(n_cosine, T, CONFIG["TR"])
    nominal = CONFIG["COSINE_CUTOFF_S_NOMINAL"]
    if n_cosine and not (p_low < nominal <= p_high):
        log.warning("%s: %d регрессоров cosine* отвечают периоду отсечения "
                    "в (%.0f, %.0f] с, номинал %.0f с вне интервала",
                    sub, n_cosine, p_low, p_high, nominal)
    rec.update(n_cosine=n_cosine,
               cosine_cutoff_low_s=float(p_low),
               cosine_cutoff_high_s=float(p_high))

    fd = conf["framewise_displacement"].to_numpy(dtype=float)
    _, censor = build_design(conf, src["conf_json"], gsr=False)
    rec.update(fd_mean=float(np.nanmean(fd)),
               fd_median=float(np.nanmedian(fd)),
               fd_max=float(np.nanmax(fd)),
               fd_mean_keep=float(np.nanmean(fd[~censor])),
               dvars_mean=float(np.nanmean(conf["std_dvars"].to_numpy(dtype=float))))
    log.info("%s: готово, %d объёмов из %d, отброшено областей: %s", sub,
             rec["n_keep"], T,
             ", ".join(f"{s['name']} {rec['n_dropped_roi_' + s['name']]}"
                       for s in specs))
    return rec


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Экспорт для замера зависимости баркода от длины окна (ветвь зигзага)
# --------------------------------------------------------------------------- #

PHENO_COLS = {                      # наше имя -> имя в файле ABIDE
    "dx_group": "DX_GROUP",         # 1 = ASD, 2 = TD
    "age": "AGE_AT_SCAN",
    "sex": "SEX",                   # 1 = M, 2 = F
    "fiq": "FIQ",
    "viq": "VIQ",
}


def load_phenotype(path: Path, subjects: list) -> pd.DataFrame:
    """Строки фенотипа ABIDE для заданных sub-XXXXXXX, в том же порядке.

    Сопоставление идёт по числовому SUB_ID, извлечённому из имени каталога.
    Отсутствующие колонки не выдумываются: заполняются NaN с предупреждением,
    чтобы потребитель видел пропуск, а не правдоподобное значение.
    """
    ph = pd.read_csv(path)
    ph.columns = [c.strip() for c in ph.columns]
    if "SUB_ID" not in ph.columns:
        raise RuntimeError(f"В {path.name} нет колонки SUB_ID")
    ph = ph.set_index(ph["SUB_ID"].astype(int))

    sid = [int(re.sub(r"\D", "", s)) for s in subjects]
    missing = [s for s, i in zip(subjects, sid) if i not in ph.index]
    if missing:
        raise RuntimeError(f"Нет строк фенотипа: {missing}")

    out = pd.DataFrame(index=range(len(subjects)))
    out["subject"] = subjects
    for ours, theirs in PHENO_COLS.items():
        if theirs in ph.columns:
            out[ours] = [ph.at[i, theirs] for i in sid]
        else:
            log.warning("В фенотипе нет колонки %s, ставлю NaN", theirs)
            out[ours] = np.nan
    return out


def common_node_set(out: Path, parcellation: str, variant: str) -> tuple:
    """Общий набор узлов по правилу полной выборки: область, потерянная хотя бы
    у одного обработанного испытуемого, исключается у всех.

    Набор берётся из coverage_summary.tsv, то есть по ВСЕМ обработанным
    испытуемым, а не по экспортируемой подвыборке. Иначе подвыборка из десяти
    человек дала бы свой, более широкий набор узлов, и он молча разошёлся бы
    с набором основной ветви.
    """
    cov_path = out / "coverage_summary.tsv"
    if not cov_path.exists():
        raise RuntimeError("Нет coverage_summary.tsv: сначала полный прогон "
                           "постобработки, затем экспорт")
    cov = pd.read_csv(cov_path, sep="\t")
    cov = cov[cov.parcellation == parcellation]
    if cov.empty:
        raise RuntimeError(f"В coverage_summary.tsv нет парцелляции {parcellation}")
    n_cov_subjects = int(cov.n_subjects.max())

    labels = pd.read_csv(out / variant / parcellation / "labels.tsv",
                         sep="\t").label.tolist()
    dropped = set(cov.loc[cov.n_dropped > 0, "label"])
    keep = [i for i, l in enumerate(labels) if l not in dropped]
    kept_labels = [labels[i] for i in keep]
    return np.asarray(keep, dtype=int), kept_labels, n_cov_subjects


def load_centroids(path: Path | None, labels: list) -> np.ndarray:
    """Координаты центроидов в порядке labels; NaN, если файла нет."""
    if path is None or not path.exists():
        log.warning("Центроиды не переданы, поле centroids заполнено NaN")
        return np.full((len(labels), 3), np.nan, dtype=np.float32)
    df = pd.read_csv(path, sep="\t")
    lab_col = next((c for c in df.columns if c.lower() in ("label", "name", "roi")),
                   None)
    xyz = [c for c in df.columns if c.lower() in ("x", "y", "z")]
    if lab_col is None or len(xyz) != 3:
        log.warning("Не разобрал %s (нужны колонки label и x,y,z), ставлю NaN",
                    path.name)
        return np.full((len(labels), 3), np.nan, dtype=np.float32)
    df = df.set_index(lab_col)
    C = np.full((len(labels), 3), np.nan, dtype=np.float32)
    for i, l in enumerate(labels):
        if l in df.index:
            C[i] = df.loc[l, ["x", "y", "z"]].to_numpy(dtype=float)
    n_missing = int(np.isnan(C[:, 0]).sum())
    if n_missing:
        log.warning("Центроиды не найдены для %d областей", n_missing)
    return C


def select_window_scan_subjects(qc: pd.DataFrame, pheno: pd.DataFrame,
                                n_total: int, n_zero_censor: int,
                                n_high_censor: int) -> list:
    """Отбор подвыборки для замера окна. Детерминированный, без генератора.

    Правила, в порядке применения:
      1. только проходящие первичный критерий (медиана FD <= 0.14, n_keep >= 120);
      2. n_zero_censor человек с нулём цензурированных объёмов — чистая база,
         где замер не загрязнён разрывом ряда;
      3. n_high_censor человек с наибольшей долей цензурированных среди
         оставшихся — на них проверяется, ломает ли разрыв результат;
      4. остальные добираются так, чтобы разброс по медианному FD был
         максимальным (берутся крайние и срединные ранги);
      5. группы уравниваются: поровну ASD и TD, насколько позволяет отбор.

    Балансировка групп идёт последней и может сместить состав по пунктам 2-4:
    это сознательный размен, и при нехватке кандидатов выводится предупреждение,
    а не тихая подмена.
    """
    df = qc.merge(pheno, on="subject", how="inner")
    ok = (df.fd_median <= 0.14) & (df.n_keep >= 120)
    df = df[ok].copy()
    if df.empty:
        raise RuntimeError("Первичный критерий не проходит никто")
    df["pct_censored"] = 100.0 * df.n_censored / df.n_volumes

    picked: list = []

    def take(sub_df, k):
        for s in sub_df.subject.tolist():
            if len(picked) >= n_total:
                break
            if s not in picked:
                picked.append(s)
                k -= 1
                if k <= 0:
                    break

    zero = df[df.n_censored == 0].sort_values(["fd_median", "subject"])
    if len(zero) < n_zero_censor:
        log.warning("С нулём цензурированных объёмов только %d, требовалось %d",
                    len(zero), n_zero_censor)
    take(zero, n_zero_censor)

    high = df[df.n_censored > 0].sort_values(["pct_censored", "subject"],
                                             ascending=[False, True])
    if len(high) < n_high_censor:
        log.warning("С ненулевым цензурированием только %d, требовалось %d",
                    len(high), n_high_censor)
    take(high, n_high_censor)

    rest = df[~df.subject.isin(picked)].sort_values(["fd_median", "subject"])
    if len(rest):
        idx = np.unique(np.linspace(0, len(rest) - 1,
                                    max(n_total - len(picked), 1)).astype(int))
        take(rest.iloc[idx], n_total - len(picked))

    sel = df[df.subject.isin(picked)]
    n_asd = int((sel.dx_group == 1).sum())
    n_td = int((sel.dx_group == 2).sum())
    if abs(n_asd - n_td) > 1:
        log.warning("Состав по группам %d ASD / %d TD — не уравнен; "
                    "кандидатов не хватило", n_asd, n_td)
    return sorted(picked)


def export_window_scan(out: Path, parcellation: str, variant: str,
                       subjects: list, pheno_path: Path,
                       centroids_path: Path | None,
                       expect_nodes: int | None) -> Path:
    """Собирает window_scan_input.npz: ряды до отбрасывания, маска, ковариаты.

    Ковариаты лежат в том же порядке, что ряды, — как в connectivity.npz.
    Слияние по идентификатору на стороне потребителя не требуется и потому не
    может рассогласоваться.
    """
    ts_dir = out / variant / parcellation / "timeseries"
    keep_idx, labels, n_cov_subjects = common_node_set(out, parcellation, variant)
    log.info("Общий набор узлов: %d из %d (правило полной выборки, %d испытуемых "
             "в coverage_summary)", len(labels), len(keep_idx) + 0, n_cov_subjects)
    if expect_nodes is not None and len(labels) != expect_nodes:
        raise RuntimeError(f"Узлов {len(labels)}, ожидалось {expect_nodes}. "
                           f"Либо обработана не вся выборка, либо изменился "
                           f"порог покрытия")

    qc = pd.read_csv(out / "qc_postproc.tsv", sep="\t")
    qc = qc.set_index("subject")

    TS, CEN = [], []
    for sub in subjects:
        f_full = ts_dir / f"{sub}_ts_full.npy"
        f_cen = ts_dir / f"{sub}_censor.npy"
        if not f_full.exists():
            raise RuntimeError(f"Нет {f_full.name}. Ряды до отбрасывания "
                               f"появляются только при SAVE_UNCENSORED=True; "
                               f"нужен повторный прогон с --overwrite")
        Y = np.load(f_full)[:, keep_idx]
        c = np.load(f_cen)
        if len(Y) != len(c):
            raise RuntimeError(f"{sub}: {len(Y)} объёмов против {len(c)} в маске")
        if np.isnan(Y).any():
            raise RuntimeError(f"{sub}: NaN в общем наборе узлов — набор "
                               f"построен неверно")
        TS.append(Y.astype(np.float32))
        CEN.append(c.astype(bool))

    n_vol = {len(y) for y in TS}
    if len(n_vol) != 1:
        raise RuntimeError(f"Разное число объёмов у испытуемых: {sorted(n_vol)}")

    pheno = load_phenotype(pheno_path, subjects)
    cols = {k: pheno[k].to_numpy() for k in PHENO_COLS}
    for k in ("fd_mean", "fd_median", "fd_mean_keep", "n_keep", "n_censored",
              "n_cosine", "cosine_cutoff_low_s", "cosine_cutoff_high_s"):
        if k in qc.columns:
            cols[k] = np.asarray([qc.at[s, k] for s in subjects])
        else:
            log.warning("В qc_postproc.tsv нет колонки %s", k)

    T = next(iter(n_vol))
    meta = dict(
        parcellation=parcellation, variant=variant,
        tr=CONFIG["TR"], dummy=CONFIG["DUMMY"], n_volumes=T,
        lowpass_hz=CONFIG["LOWPASS_HZ"], filter_order=CONFIG["FILTER_ORDER"],
        interp_at_censored=variant.endswith("_bp"),
        censor_source=CONFIG["CENSOR_SOURCE"],
        coverage_min=CONFIG["COVERAGE_MIN"],
        cosine_cutoff_s_nominal=CONFIG["COSINE_CUTOFF_S_NOMINAL"],
        n_subjects_in_coverage=n_cov_subjects,
        exclusion_primary=CONFIG["EXCLUSION_PRIMARY"],
    )

    dst = out / "window_scan_input.npz"
    np.savez_compressed(
        dst,
        ts=np.stack(TS), censor=np.stack(CEN),
        subject_ids=np.asarray(subjects),
        node_names=np.asarray(labels),
        centroids=load_centroids(centroids_path, labels),
        meta_json=json.dumps(meta, ensure_ascii=False),
        **cols,
    )

    p_low = float(np.nanmin(cols.get("cosine_cutoff_low_s", [np.nan])))
    p_high = float(np.nanmax(cols.get("cosine_cutoff_high_s", [np.nan])))
    log.info("Записано %s: %d испытуемых x %d объёмов x %d узлов",
             dst, len(subjects), T, len(labels))
    if np.isfinite(p_low):
        log.info("Период отсечения по числу cosine*: (%.0f, %.0f] с, "
                 "то есть минимальное окно %.0f-%.0f TR",
                 p_low, p_high, p_low / CONFIG["TR"], p_high / CONFIG["TR"])
    return dst


# --------------------------------------------------------------------------- #

def summarize_coverage(out: Path) -> None:
    files = sorted((out / "coverage").glob("*_coverage.tsv"))
    if not files:
        return
    cov = pd.concat([pd.read_csv(p, sep="\t") for p in files])
    agg = (cov.groupby(["parcellation", "label"])
              .agg(n_dropped=("dropped", "sum"), n_subjects=("dropped", "size"),
                   mean_coverage=("coverage", "mean"),
                   n_vox_atlas=("n_vox_atlas", "first"))
              .reset_index())
    agg["pct_dropped"] = (100 * agg.n_dropped / agg.n_subjects).round(1)
    agg = agg.sort_values(["parcellation", "n_dropped"], ascending=[True, False])
    agg.to_csv(out / "coverage_summary.tsv", sep="\t", index=False)
    worst = agg[agg.n_dropped > 0]
    log.info("Областей, теряемых хотя бы у одного испытуемого: %d", len(worst))
    for _, r in worst.head(20).iterrows():
        log.info("   %-28s теряется у %5.1f%% испытуемых, среднее покрытие %.2f, "
                 "вокселей в атласе %d",
                 r.label, r.pct_dropped, r.mean_coverage, r.n_vox_atlas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derivatives", type=Path,
                    help="каталог производных fMRIPrep (не нужен при "
                         "--export-window-scan)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--atlas-dir", type=Path,
                    help="каталог атласов (не нужен при --export-window-scan)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")

    g = ap.add_argument_group("экспорт для замера окна (ветвь зигзага)")
    g.add_argument("--export-window-scan", action="store_true",
                   help="собрать window_scan_input.npz по уже посчитанной "
                        "постобработке и выйти")
    g.add_argument("--phenotype", type=Path,
                   help="Phenotypic_V1_0b_preprocessed1.csv")
    g.add_argument("--centroids", type=Path,
                   help="centroids_aal3.tsv из qc_fc.py, необязательно")
    g.add_argument("--variant", default=None,
                   help="вариант подавления шума; по умолчанию основной")
    g.add_argument("--parcellation", default=None,
                   help="парцелляция; по умолчанию основная")
    g.add_argument("--n-select", type=int, default=10,
                   help="сколько испытуемых отобрать, если --subjects не задан")
    g.add_argument("--n-zero-censor", type=int, default=3)
    g.add_argument("--n-high-censor", type=int, default=3)
    g.add_argument("--expect-nodes", type=int, default=None,
                   help="ожидаемое число узлов общего набора; расхождение — "
                        "ошибка, а не предупреждение")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if a.export_window_scan:
        if a.phenotype is None:
            log.error("--export-window-scan требует --phenotype")
            return 1
        variant = a.variant or CONFIG["MAIN_VARIANT"]
        parcellation = a.parcellation or CONFIG["MAIN_PARCELLATION"]
        subs = a.subjects
        if not subs:
            qc = pd.read_csv(a.out / "qc_postproc.tsv", sep="\t")
            pheno = load_phenotype(a.phenotype, qc.subject.tolist())
            subs = select_window_scan_subjects(
                qc, pheno, a.n_select, a.n_zero_censor, a.n_high_censor)
            log.info("Отобрано %d испытуемых: %s", len(subs), ", ".join(subs))
        export_window_scan(a.out, parcellation, variant, subs,
                           a.phenotype, a.centroids, a.expect_nodes)
        return 0

    if a.derivatives is None or a.atlas_dir is None:
        log.error("Нужны --derivatives и --atlas-dir")
        return 1

    subs = a.subjects or sorted(p.name for p in a.derivatives.glob("sub-*")
                                if p.is_dir())
    if not subs:
        log.error("В %s не найдено каталогов sub-*", a.derivatives)
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
        indent=2, ensure_ascii=False), encoding="utf-8")

    done = a.out / "qc_postproc.tsv"
    already = set()
    if done.exists() and not a.overwrite:
        already = set(pd.read_csv(done, sep="\t").subject)
        subs = [s for s in subs if s not in already]
        log.info("Пропускаю %d уже обработанных", len(already))

    recs, failed = [], []
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            fut = {ex.submit(process_subject, s, a.derivatives, a.out,
                             a.atlas_dir): s for s in subs}
            for f in as_completed(fut):
                s = fut[f]
                try:
                    recs.append(f.result())
                except Exception as e:                       # noqa: BLE001
                    log.error("%s: ОШИБКА %s", s, e)
                    failed.append((s, str(e)))
    else:
        for s in subs:
            try:
                recs.append(process_subject(s, a.derivatives, a.out, a.atlas_dir))
            except Exception as e:                           # noqa: BLE001
                log.error("%s: ОШИБКА %s", s, e)
                failed.append((s, str(e)))

    if recs:
        df = pd.DataFrame(recs).sort_values("subject")
        if already and done.exists():
            df = pd.concat([pd.read_csv(done, sep="\t"), df]).sort_values("subject")
        df.to_csv(done, sep="\t", index=False)
        log.info("Сводка QC: %s (%d строк)", done, len(df))
        summarize_coverage(a.out)

    if failed:
        pd.DataFrame(failed, columns=["subject", "error"]).to_csv(
            a.out / "failed.tsv", sep="\t", index=False)
        log.error("Не обработано: %d, список в failed.tsv", len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

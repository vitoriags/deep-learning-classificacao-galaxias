import shutil
import zipfile
from pathlib import Path

import tensorflow as tf


def train_test_validation_split(diretorio: str, tamanhoImg: int):
    print(diretorio)

    if str(diretorio).endswith("/test"):
        # Se o diretório for de teste, não há rótulos
        ds = tf.keras.utils.image_dataset_from_directory(
            diretorio,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=5,
            image_size=(tamanhoImg, tamanhoImg),
            shuffle=False,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
        )

    else:
        ds = tf.keras.utils.image_dataset_from_directory(
            diretorio,
            labels="inferred",
            label_mode="categorical",  # categorical: [0, 0, 1] | binary: [0, 1, 1]
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(tamanhoImg, tamanhoImg),
            shuffle=True,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
        )
    return ds


def extrair_zip(zip_path: Path, destino_base: Path):
    nome_pasta_zip = zip_path.stem  # Ex: "segmented"
    destino_final = destino_base / nome_pasta_zip

    if destino_final.exists() and any(destino_final.iterdir()):
        print(f"{zip_path.name} já foi extraído. Pulando extração.\n")
        return

    destino_final.mkdir(parents=True, exist_ok=True)

    print(f"Extraindo {zip_path.name} para {destino_final}...\n")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files = zip_ref.namelist()
        total = len(files)

        for i, file in enumerate(files, 1):
            zip_ref.extract(file, destino_base)  # mantém estrutura original do zip
            print(f"[{i}/{total}] Extraído: {file}")

    print("Extração concluída.\n")


def moverpastas(pasta: str):
    shutil.move(f"/content/{pasta}/train", "/content/")
    shutil.move(f"/content/{pasta}/validation", "/content/")
    shutil.move(f"/content/{pasta}/test", "/content/")
    shutil.rmtree(pasta)


def contagemimagens(RotuloImagensTest, folder):
    BASE_DIR = Path(__file__).resolve().parent.parent

    pastas = [
        BASE_DIR / "data" / "images" / folder / "train",
        BASE_DIR / "data" / "images" / folder / "validation",
    ]

    soma = 0
    listatv = 0

    for pasta in pastas:
        print()
        print(pasta)

        lista0 = len(list((pasta / "0").glob("*")))
        lista1 = len(list((pasta / "1").glob("*")))

        porcentagem = (lista0 * 100) / (lista0 + lista1)
        print(f"0 | {lista0} | {porcentagem:.1f}%")

        porcentagem = (lista1 * 100) / (lista0 + lista1)
        print(f"1 | {lista1} | {porcentagem:.1f}%")

        listatrainval = lista0 + lista1
        porcentagem = (listatrainval * 100) / 239029
        print(f"t | {listatrainval} | {porcentagem:.1f}%")

        listatv += listatrainval

    print()

    lista0 = sum(1 for x in RotuloImagensTest if x == 0)
    lista1 = sum(1 for x in RotuloImagensTest if x == 1)

    print(BASE_DIR / "data" / "images" / folder / "test")
    porcentagem = (lista0 * 100) / (lista0 + lista1)
    print(f"0 | {lista0} | {porcentagem:.1f}%")

    porcentagem = (lista1 * 100) / (lista0 + lista1)
    print(f"1 | {lista1} | {porcentagem:.1f}%")

    lista = lista0 + lista1
    porcentagem = (lista * 100) / 239029
    print(f"t | {lista} | {porcentagem:.1f}%")

    print()
    soma = listatv + lista
    print(f"total: {soma}")

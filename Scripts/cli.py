
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .cluster import PokerHandClusterer
from .data import EncodedDataset, load_poker_dataset, split_dataset

def _format_metrics(title: str, metrics: Dict[str, Optional[float]]) -> str:
    display = {key: (value if value is None else round(float(value), 6)) for key, value in metrics.items()}
    return f"{title}:\n" + json.dumps(display, indent=2, ensure_ascii=False)

def _extract_subset(encoded: EncodedDataset, indices: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    subset: Dict[str, Optional[np.ndarray]] = {
        "features": encoded.features[indices],
    }
    if encoded.labels is not None:
        subset["labels"] = encoded.labels[indices]
    return subset

def train_on_full_dataset(
    csv_path: Path,
    *,
    label_column: Optional[str],
    drop_columns: Optional[list[str]],
    n_clusters: Optional[int],
    random_state: int,
    model_output_path: Optional[Path] = None,
) -> None:
    Treina o modelo usando 100% dos dados disponíveis no CSV.
    
    Parameters
    ----------
    csv_path : Path
        Caminho para o arquivo CSV com os dados de treinamento.
    label_column : Optional[str]
        Nome da coluna que contém os rótulos das mãos de poker.
    drop_columns : Optional[list[str]]
        Colunas a serem removidas antes do treinamento.
    n_clusters : Optional[int]
        Número de clusters para o K-Means.
    random_state : int
        Seed para reprodutibilidade.
    model_output_path : Optional[Path]
        Caminho onde o modelo treinado será salvo.
    encoded = load_poker_dataset(
        csv_path,
        label_column=label_column,
        drop_columns=drop_columns,
    )

    clusterer = PokerHandClusterer(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(encoded.features)

    if encoded.labels is not None:
        clusterer.build_label_map(encoded.features, encoded.labels)

    metrics = clusterer.evaluate(
        encoded.features,
        labels=encoded.labels,
    )

    report = {
        "inertia": metrics.inertia,
        "silhouette": metrics.silhouette,
        "accuracy": metrics.accuracy,
    }
    print(_format_metrics("Resultados usando 100% do conjunto", report))
    
    if model_output_path:
        clusterer.save_model(model_output_path)
        print(f"\n✅ Modelo salvo em: {model_output_path}")

def train_with_split(
    csv_path: Path,
    *,
    label_column: Optional[str],
    drop_columns: Optional[list[str]],
    n_clusters: Optional[int],
    train_size: float,
    random_state: int,
    stratify_column: Optional[str],
    output_train_path: Optional[Path],
    output_test_path: Optional[Path],
    model_output_path: Optional[Path] = None,
) -> None:
    Treina o modelo dividindo os dados em conjunto de treino e teste.
    
    Parameters
    ----------
    csv_path : Path
        Caminho para o arquivo CSV com os dados.
    label_column : Optional[str]
        Nome da coluna que contém os rótulos das mãos de poker.
    drop_columns : Optional[list[str]]
        Colunas a serem removidas antes do treinamento.
    n_clusters : Optional[int]
        Número de clusters para o K-Means.
    train_size : float
        Proporção dos dados para treino (ex: 0.7 = 70%).
    random_state : int
        Seed para reprodutibilidade.
    stratify_column : Optional[str]
        Coluna para estratificação do split.
    output_train_path : Optional[Path]
        Caminho para salvar o CSV de treino.
    output_test_path : Optional[Path]
        Caminho para salvar o CSV de teste.
    model_output_path : Optional[Path]
        Caminho onde o modelo treinado será salvo.
    train_df, test_df = split_dataset(
        csv_path,
        train_size=train_size,
        random_state=random_state,
        stratify_column=stratify_column,
        output_train_path=output_train_path,
        output_test_path=output_test_path,
    )

    encoded = load_poker_dataset(
        csv_path,
        label_column=label_column,
        drop_columns=drop_columns,
    )

    train_subset = _extract_subset(encoded, train_df.index.to_numpy())
    test_subset = _extract_subset(encoded, test_df.index.to_numpy())

    clusterer = PokerHandClusterer(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(train_subset["features"])

    if train_subset.get("labels") is not None:
        clusterer.build_label_map(train_subset["features"], train_subset["labels"])

    train_metrics = clusterer.evaluate(
        train_subset["features"],
        labels=train_subset.get("labels"),
    )
    train_report = {
        "inertia": train_metrics.inertia,
        "silhouette": train_metrics.silhouette,
        "accuracy": train_metrics.accuracy,
    }
    print(_format_metrics("Resultados no conjunto de treino", train_report))

    predictions_test = clusterer.predict(test_subset["features"])
    distances = clusterer.model.transform(test_subset["features"])
    selected_distances = distances[np.arange(len(predictions_test)), predictions_test]
    test_inertia = float(np.sum(np.square(selected_distances)))

    test_silhouette = None
    if len(test_subset["features"]) > 1 and len(np.unique(predictions_test)) > 1:
        from sklearn.metrics import silhouette_score

        test_silhouette = float(
            silhouette_score(test_subset["features"], predictions_test, metric="euclidean")
        )

    test_accuracy = None
    if test_subset.get("labels") is not None and clusterer.cluster_label_map:
        predicted_labels_test = clusterer.predict_labels(test_subset["features"])
        test_accuracy = float(np.mean(predicted_labels_test == test_subset["labels"]))

    test_report = {
        "inertia": test_inertia,
        "silhouette": test_silhouette,
        "accuracy": test_accuracy,
    }
    print(_format_metrics("Resultados no conjunto de teste", test_report))
    
    if model_output_path:
        clusterer.save_model(model_output_path)
        print(f"\n✅ Modelo salvo em: {model_output_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treine um agrupador K-Means para o dataset de mãos de poker.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Caminho para o arquivo CSV contendo o dataset.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Nome da coluna que representa o rótulo da mão de poker (opcional).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Colunas adicionais que devem ser removidas antes do treinamento.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Número de clusters usados pelo K-Means (por padrão será inferido).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente para a aleatoriedade do algoritmo.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Caminho para salvar o modelo treinado (.pkl).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser(
        "full",
        help="Treina usando 100% dos dados disponíveis.",
    )
    full_parser.set_defaults(command_handler="full")

    split_parser = subparsers.add_parser(
        "split",
        help="Divide o dataset em treino e teste (70/30 por padrão) e treina o modelo.",
    )
    split_parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Proporção de dados usados para treino (padrão 0.7).",
    )
    split_parser.add_argument(
        "--stratify-column",
        type=str,
        default=None,
        help="Coluna usada para estratificar o split (recomendado informar o rótulo).",
    )
    split_parser.add_argument(
        "--output-train",
        type=Path,
        default=None,
        help="Arquivo CSV onde o conjunto de treino será salvo.",
    )
    split_parser.add_argument(
        "--output-test",
        type=Path,
        default=None,
        help="Arquivo CSV onde o conjunto de teste será salvo.",
    )
    split_parser.set_defaults(command_handler="split")

    return parser

def main(argv: Optional[list[str]] = None) -> None:
    Função principal da interface de linha de comando.
    Treina modelos de clustering K-Means em datasets de poker a partir de arquivos CSV.
    parser = build_parser()
    args = parser.parse_args(argv)

    command_handler = args.command_handler
    csv_path: Path = args.csv_path
    label_column: Optional[str] = args.label_column
    drop_columns: Optional[list[str]] = args.drop_columns
    n_clusters: Optional[int] = args.clusters
    random_state: int = args.random_state
    model_output_path: Optional[Path] = args.model_output

    if command_handler == "full":
        train_on_full_dataset(
            csv_path,
            label_column=label_column,
            drop_columns=drop_columns,
            n_clusters=n_clusters,
            random_state=random_state,
            model_output_path=model_output_path,
        )
    elif command_handler == "split":
        train_with_split(
            csv_path,
            label_column=label_column,
            drop_columns=drop_columns,
            n_clusters=n_clusters,
            train_size=args.train_size,
            random_state=random_state,
            stratify_column=args.stratify_column,
            output_train_path=args.output_train,
            output_test_path=args.output_test,
            model_output_path=model_output_path,
        )
    else:
        parser.error("Nenhum comando válido foi informado.")

if __name__ == "__main__":
    main()
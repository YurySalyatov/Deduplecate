import torch
import argparse
import sys


def parse_args():
    """Парсер аргументов командной строки с default значениями"""
    parser = argparse.ArgumentParser(
        description='Обучение нейронной сети для обнаружения дубликатов в гетерогенных графах',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument(
        '--root',
        type=str,
        default='./',
        help='Путь к JSON файлу с данными'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Директория для сохранения результатов и моделей'
    )

    # Model parameters
    parser.add_argument(
        '--hidden-channels',
        type=int,
        default=128,
        help='Размер скрытого слоя нейронной сети'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Количество слоев в нейронной сети'
    )

    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=64,
        help='Размерность эмбеддингов вершин'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Вероятность dropout'
    )

    # Training parameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Скорость обучения'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2 регуляризация'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Количество эпох обучения'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Размер батча'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Количество эпох для ранней остановки'
    )

    # Graph parameters
    parser.add_argument(
        '--num-duplicates',
        type=int,
        default=3,
        help='Количество дубликатов для каждой вершины (k)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Доля тренировочных данных'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Доля валидационных данных'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Доля тестовых данных'
    )

    # Runtime parameters
    parser.add_argument(
        '--device',
        type=int,
        default=-1,
        help='Устройство для вычислений (-1 is cpu)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed для воспроизводимости'
    )

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Интервал логирования (в эпохах)'
    )

    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Сохранять обученную модель'
    )

    parser.add_argument(
        '--no-save-model',
        action='store_false',
        dest='save_model',
        help='Не сохранять обученную модель'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Создавать визуализации'
    )

    # Experimental parameters
    parser.add_argument(
        '--model-type',
        type=str,
        default='HGT',
        choices=['HGT', 'GCN', 'GAT', 'GraphSAGE'],
        help='Тип модели графовой нейронной сети'
    )

    parser.add_argument(
        '--negative-samples-ratio',
        type=float,
        default=1.0,
        help='Соотношение негативных сэмплов к позитивным'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='roberta-base-nli-stsb-mean-tokens',
        help='Name of bert model'
    )

    args = parser.parse_args()

    # Автоматическое определение устройства
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device != -1 else 'cpu')

    # Проверка соотношений разделения данных
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️ Внимание: сумма train_ratio, val_ratio и test_ratio равна {total_ratio:.2f}, нормализую до 1.0")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    return args


def print_args(args):
    """Печать всех аргументов"""
    print("🎯 Конфигурация запуска:")
    print("=" * 50)
    for arg in vars(args):
        print(f"  {arg:25} = {getattr(args, arg)}")
    print("=" * 50)

if __name__ == "__main__":
    # Тестирование парсера
    print_args(parse_args())
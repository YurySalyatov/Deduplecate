#!/usr/bin/env python3
"""
Скрипт для установки всех зависимостей проекта
"""

import subprocess
import sys
# import os
import platform
import torch


def check_python_version():
    """Проверка версии Python"""
    print("🔍 Проверка версии Python...")
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def get_pytorch_install_command():
    """Получение команды для установки PyTorch в зависимости от платформы"""
    system = platform.system().lower()

    # Проверяем наличие CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"🎯 Обнаружена CUDA {cuda_version}")
        return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
    else:
        print("ℹ️  CUDA не обнаружена, будет установлен CPU-вариант PyTorch")
        if system == "windows":
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        else:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"


def install_requirements():
    """Установка зависимостей из requirements.txt"""
    print("\n📦 Установка зависимостей из requirements.txt...")

    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Зависимости успешно установлены")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при установке зависимостей: {e}")
        print(f"Stderr: {e.stderr}")
        return False


def install_torch_geometric():
    """Установка PyTorch Geometric с зависимостями"""
    print("\n🧠 Установка PyTorch Geometric...")

    # Получаем версию PyTorch и CUDA
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None

    print(f"PyTorch версия: {torch_version}")
    print(f"CUDA доступна: {cuda_available}")
    if cuda_available:
        print(f"CUDA версия: {cuda_version}")

    # Определяем версию для установки
    if cuda_available and cuda_version:
        # Для CUDA
        cuda_suffix = f"cu{cuda_version.replace('.', '')}"
        torch_geo_packages = [
            f"torch-scatter -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html",
            f"torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html",
            f"torch-cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html",
            f"torch-spline-conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html",
            "torch-geometric"
        ]
    else:
        # Для CPU
        torch_geo_packages = [
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
            "torch-geometric"
        ]

    # Устанавливаем пакеты PyTorch Geometric
    for package in torch_geo_packages:
        print(f"📥 Установка {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True)
            print(f"✅ {package.split()[0]} успешно установлен")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при установке {package}: {e}")
            return False

    return True


def verify_installation():
    """Проверка успешности установки"""
    print("\n🔍 Проверка установки...")

    try:
        import torch
        import torch_geometric
        import numpy as np
        import sklearn
        import pandas as pd

        print("✅ PyTorch:", torch.__version__)
        print("✅ PyTorch Geometric:", torch_geometric.__version__)
        print("✅ NumPy:", np.__version__)
        print("✅ scikit-learn:", sklearn.__version__)
        print("✅ pandas:", pd.__version__)

        # Проверка доступности CUDA
        print(f"🎯 CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 CUDA версия: {torch.version.cuda}")
            print(f"🎯 GPU устройств: {torch.cuda.device_count()}")

        return True

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


def main():
    """Основная функция установки"""
    print("🚀 Начало установки зависимостей для проекта обнаружения дубликатов в графах")
    print("=" * 80)

    # Проверка версии Python
    if not check_python_version():
        sys.exit(1)

    # Обновление pip
    print("\n🔄 Обновление pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ pip успешно обновлен")
    except subprocess.CalledProcessError:
        print("⚠️  Не удалось обновить pip, продолжаем установку...")

    # Установка PyTorch
    print("\n🔥 Установка PyTorch...")
    pytorch_command = get_pytorch_install_command()
    try:
        subprocess.run(pytorch_command.split(), check=True)
        print("✅ PyTorch успешно установлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при установке PyTorch: {e}")
        sys.exit(1)

    # Установка PyTorch Geometric
    if not install_torch_geometric():
        print("❌ Не удалось установить PyTorch Geometric")
        sys.exit(1)

    # Установка остальных зависимостей
    if not install_requirements():
        print("❌ Не удалось установить все зависимости")
        sys.exit(1)

    # Проверка установки
    if verify_installation():
        print("\n🎉 Все зависимости успешно установлены и проверены!")
        print("\n📚 Доступные команды:")
        print("  python main.py          - запуск основного скрипта")
        print("  jupyter notebook        - запуск Jupyter Notebook")
        print("  python -c \"import torch_geometric; print('OK')\" - проверка PyG")
    else:
        print("\n⚠️  Возникли проблемы при установке некоторых зависимостей")
        sys.exit(1)


if __name__ == "__main__":
    main()
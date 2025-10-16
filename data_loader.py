import json
import os.path
from typing import Dict

import torch


class DataLoader:
    def __init__(self, data_path: str):
        self.test_data = None
        self.train_data = None
        self.data_path = os.path.join(data_path, 'na-data-kdd18', 'data', 'global')
        self.raw_data = None

    def load_train_data(self):
        with open(os.path.join(self.data_path, "name_to_pubs_train_500.json"), 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)

    def load_test_data(self):
        with open(os.path.join(self.data_path, "name_to_pubs_test_100.json"), 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

    def load_data(self) -> Dict:
        """Загрузка данных из JSON файла"""
        with open(os.path.join(self.data_path, "pubs_raw.json"), 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        return self.raw_data

    def load_train_idx(self) -> list:
        unique_authors = self.get_entity_mappings(self.get_unique_entities())['author']
        train_author_idx = []
        for name in self.train_data:
            author_data = self.train_data[name]
            for id in author_data:
                train_author_idx.append(unique_authors[id])
        return train_author_idx

    def load_tset_idx(self) -> list:
        unique_authors = self.get_entity_mappings(self.get_unique_entities())['author']
        test_author_idx = []
        for name in self.test_data:
            author_data = self.test_data[name]
            for id in author_data:
                test_author_idx.append(unique_authors[id])
        return test_author_idx

    def get_unique_entities(self) -> Dict[str, set]:
        """Извлечение уникальных сущностей из данных"""
        if self.raw_data is None:
            self.load_data()

        authors = set()
        papers = set()
        organizations = set()
        venues = set()

        for paper_id in self.raw_data:
            data = self.raw_data[paper_id]
            papers.add(paper_id)

            venue = data.get('venue')
            if venue:
                venues.add(venue)

            for author in data.get('authors', []):
                author_id = author['id']
                authors.add(author_id)

                org = author.get('org')
                if org:
                    organizations.add(org)

        return {
            'author': authors,
            'paper': papers,
            'organization': organizations,
            'venue': venues
        }

    def get_entity_mappings(self, entities: Dict[str, set]) -> Dict[str, Dict[str, int]]:
        """Создание mapping'ов для сущностей"""
        mappings = {}

        for entity_type, entity_set in entities.items():
            mappings[entity_type] = {entity: idx for idx, entity in enumerate(sorted(entity_set))}

        return mappings

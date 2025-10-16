import torch
from torch_geometric.data import HeteroData
from data_loader import DataLoader
import os


class GraphBuilder:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.entity_mappings = None
        self.original_data = None

    def build_hetero_data(self) -> HeteroData:
        """Построение гетерогенного графа из исходных данных"""
        data = HeteroData()

        # Загрузка данных и создание mapping'ов
        raw_data = self.data_loader.load_data()
        entities = self.data_loader.get_unique_entities()
        self.entity_mappings = self.data_loader.get_entity_mappings(entities)

        # Создание узлов
        for entity_type, mapping in self.entity_mappings.items():
            num_nodes = len(mapping)
            data[entity_type].x = torch.arange(num_nodes).reshape(-1, 1).float()

        # Создание ребер
        author_paper_edges = []
        paper_venue_edges = []
        author_org_edges = []
        paper_text_features = [""] * len(self.entity_mappings['paper'])
        # author_text_features = [""] * len(self.entity_mappings['author'])
        venue_text_features = [""] * len(self.entity_mappings['venue'])
        organization_text_features = [""] * len(self.entity_mappings['organization'])

        for paper_id in raw_data:
            paper_data = raw_data[paper_id]
            paper_idx = self.entity_mappings['paper'][paper_id]
            text = paper_data.get('title', "") + "." + paper_data.get('abstract', "")
            if not text:
                text = "Empty text"
            # Связь paper-venue
            paper_text_features[paper_idx] = text
            venue = paper_data.get('venue')
            if venue and venue in self.entity_mappings['venue']:
                venue_idx = self.entity_mappings['venue'][venue]
                venue_text_features[venue_idx] = venue
                paper_venue_edges.append([paper_idx, venue_idx])

            for author in paper_data.get('authors', []):
                author_idx = self.entity_mappings['author'][author['id']]
                # if author_text_features[author_idx] == "":
                #     author_text_features[author_idx] = f'author_{author_idx}'
                # Связь author-paper
                author_paper_edges.append([author_idx, paper_idx])

                # Связь author-organization
                org = author.get('org')
                if org and org in self.entity_mappings['organization']:
                    org_idx = self.entity_mappings['organization'][org]
                    if organization_text_features[org_idx] == "":
                        organization_text_features[org_idx] = org
                    author_org_edges.append((author_idx, org_idx))

        # Добавление ребер в граф
        if author_paper_edges:
            edge_index = torch.tensor(author_paper_edges, dtype=torch.long).t()
            data['author', 'writes', 'paper'].edge_index = edge_index

        if paper_venue_edges:
            edge_index = torch.tensor(paper_venue_edges, dtype=torch.long).t()
            data['paper', 'published_in', 'venue'].edge_index = edge_index

        if author_org_edges:
            unique_author_org_edges = set(author_org_edges)
            edge_index = torch.tensor([[a, b] for a, b in unique_author_org_edges], dtype=torch.long).t()
            data['author', 'affiliated_with', 'organization'].edge_index = edge_index

        self.original_data = data
        data.paper_text_features = paper_text_features
        # data.author_text_features = author_text_features
        data.venue_text_features = venue_text_features
        data.organization_text_features = organization_text_features
        return data

    def create_duplicates(self, data: HeteroData, num_duplicates) -> HeteroData:
        """Создание дубликатов авторов и добавление связей между ними"""
        num_authors = data['author'].x.shape[0]

        # Создание дубликатов авторов
        total_authors = num_authors * (num_duplicates + 1)
        data['author'].x = torch.arange(total_authors).reshape(-1, 1).float()
        # text_features = data.author_text_features
        new_text_features = [""] * total_authors
        # Создание связей между оригинальными авторами и их дубликатами
        duplicate_edges = []
        org_edges = data['author', 'affiliated_with', 'organization'].edge_index
        paper_edges = data['author', 'writes', 'paper'].edge_index
        new_authors_org = torch.tensor([], dtype=torch.long).reshape(2, -1)
        for orig_author_idx in range(num_authors):
            duplicate_ind = [num_authors * dup_num + orig_author_idx for dup_num in range(num_duplicates + 1)]
            for i in range(len(duplicate_ind)):
                ind1 = duplicate_ind[i]
                new_text_features[ind1] = f"author_{ind1}"
                for j in range(i + 1, len(duplicate_ind)):
                    ind2 = duplicate_ind[j]
                    # TODO necessary duplicate edges?
                    # if ind1 == ind2:
                    #     continue
                    # Двунаправленные связи между оригиналом и дубликатом
                    duplicate_edges.append([ind1, ind2])
            org_mask = torch.isin(org_edges[0], orig_author_idx)
            probably_orgs = org_edges[:, org_mask]
            target_orgs = probably_orgs[1, torch.randint(0, probably_orgs.size(1), (num_duplicates,))]
            # TODO maybe not only 1 org, but range(1, len(orgs))?
            if target_orgs.size(0) > 0:
                new_authors_org_edges = torch.stack([torch.tensor(duplicate_ind[1:], dtype=torch.long), target_orgs])
                new_authors_org = torch.cat((new_authors_org, new_authors_org_edges), dim=1)
            author_pairs_mask = torch.isin(paper_edges[0], orig_author_idx)
            author_paper_idx = paper_edges[:, author_pairs_mask]
            for i in range(author_paper_idx.shape[1] // (num_duplicates + 1)):
                for j in range(num_duplicates + 1):
                    author_paper_idx[0, i * (num_duplicates + 1) + j] = duplicate_ind[j]
            ost_ind = 0
            for i in range((author_paper_idx.shape[1] // (num_duplicates + 1)) * (num_duplicates + 1),
                           author_paper_idx.shape[1]):
                author_paper_idx[0, i] = duplicate_ind[ost_ind]
                ost_ind += 1
            data['author', 'writes', 'paper'].edge_index[:, author_pairs_mask] = author_paper_idx
        if duplicate_edges:
            edge_index = torch.tensor(duplicate_edges, dtype=torch.long).t()
            data['author', 'duplicate', 'author'].edge_index = edge_index

            # Создание меток для ребер-дубликатов
            data['author', 'duplicate', 'author'].edge_label = torch.ones(
                edge_index.size(1), dtype=torch.float
            )
        data['author', 'affiliated_with', 'organization'].edge_index = torch.cat(
            (org_edges, new_authors_org), dim=1)
        data.author_text_features = new_text_features
        return data

    def prepare_train_data(self, data: HeteroData, train_ratio=0.64, val_ratio=0.16) -> HeteroData:
        """Подготовка данных для обучения"""
        # Здесь можно добавить negative sampling для ребер-дубликатов
        # и разделение на train/val/test
        positive_edges = data['author', 'duplicate', 'author'].edge_index.t()
        positive_labels = data['author', 'duplicate', 'author'].edge_label
        n = data['author'].x.size(0)
        all_edges = torch.cartesian_prod(torch.arange(n), torch.arange(n))
        all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
        all_edges = all_edges[all_edges[:, 0] < all_edges[:, 1]]
        correct_str = set([f"{u}_{v}" for u, v in positive_edges.tolist()])
        all_str = set([f"{u}_{v}" for u, v in all_edges.tolist()])
        mask = torch.tensor([s not in correct_str for s in all_str])
        incorrect_pool = all_edges[mask, :]
        if len(incorrect_pool) <= 0:
            raise Exception("nothing for negative edges")
        indices = torch.randperm(len(incorrect_pool))[:positive_labels.size(0)]
        negative_edges = incorrect_pool[indices]
        negative_labels = torch.zeros(negative_edges.size(0))
        data['author', 'duplicate', 'author'].edge_index = torch.cat((positive_edges, negative_edges), dim=0).t()
        data['author', 'duplicate', 'author'].edge_label = torch.cat((positive_labels, negative_labels))

        n = data['author', 'duplicate', 'author'].edge_index.size(1)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        test_size = n - train_size - val_size

        indices = torch.randperm(n)

        # Разделяем индексы
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Создаем маски
        mask_train = torch.zeros(n, dtype=torch.bool)
        mask_val = torch.zeros(n, dtype=torch.bool)
        mask_test = torch.zeros(n, dtype=torch.bool)

        mask_train[train_indices] = True
        mask_val[val_indices] = True
        mask_test[test_indices] = True

        data.train_mask = mask_train
        data.val_mask = mask_val
        data.test_mask = mask_test

        return data


def text_to_bert_embedding(author_text, paper_text, venue_text, organization_text, root, model_name, device):
    cache_path = os.path.join(root, f"aminer_text_{model_name}.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        author_embedding = data['author']
        paper_embedding = data['paper']
        venue_embedding = data['venue']
        organization_embedding = data['organization']
        return author_embedding, paper_embedding, venue_embedding, organization_embedding
    author_len = len(author_text)
    paper_len = len(paper_text)
    venue_len = len(venue_text)
    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer(model_name, cache_folder=os.path.join(root, "sbert"), device=device)
    all_text = []
    for text in author_text:
        all_text.append(text)
    for text in paper_text:
        all_text.append(text)
    for text in venue_text:
        all_text.append(text)
    for text in organization_text:
        all_text.append(text)

    embedding = bert.encode(all_text, show_progress_bar=True, convert_to_tensor=True)
    embedding = embedding.cpu()
    author_embedding = embedding[:author_len, :]
    paper_embedding = embedding[author_len:author_len + paper_len, :]
    venue_embedding = embedding[author_len + paper_len:author_len + paper_len + venue_len, :]
    organization_embedding = embedding[author_len + paper_len + venue_len:, :]
    torch.save({'author': author_embedding, 'paper': paper_embedding, 'venue': venue_embedding,
                'organization': organization_embedding}, cache_path)
    return author_embedding, paper_embedding, venue_embedding, organization_embedding

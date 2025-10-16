import os.path

from torch_geometric.data import HeteroData

from data_loader import DataLoader
from graph_builder import GraphBuilder, text_to_bert_embedding, text_to_embeddings
from model import DuplicatePredictor
from train import train_model
from args import parse_args
import torch


def get_form_hetero_graph(hetero_data, edge_name, idx):
    # 'author', 'affiliated_with', 'organization'
    author_organisation_edge_index = hetero_data[edge_name].edge_index
    organisation_mask = torch.isin(author_organisation_edge_index[0], idx)
    selected_author_organisation_edge_idx = author_organisation_edge_index[:, organisation_mask]
    selected_org_indices = torch.unique(selected_author_organisation_edge_idx[1])
    return selected_author_organisation_edge_idx, selected_org_indices


def sub_graph_for_testing(hetero_data: HeteroData, indexes):
    authors_idx = indexes

    # subgraph = HeteroData()
    hetero_data['author'].x = hetero_data['author'].x[authors_idx]

    author_paper_edge_index = hetero_data['author', 'writes', 'paper'].edge_index
    paper_mask = torch.isin(author_paper_edge_index[0], authors_idx)
    selected_author_paper_edge_idx = author_paper_edge_index[:, paper_mask]
    selected_paper_indices = torch.unique(selected_author_paper_edge_idx[1])
    hetero_data['paper'].x = hetero_data['paper'].x[selected_paper_indices]
    hetero_data.paper_text_features = [hetero_data.paper_text_features[i] for i in selected_paper_indices]

    # author_organisation_edge_index = hetero_data['author', 'affiliated_with', 'organization'].edge_index
    # organisation_mask = torch.isin(author_organisation_edge_index[0], authors_idx)
    # selected_author_organisation_edge_idx = author_organisation_edge_index[:, organisation_mask]
    # selected_org_indices = torch.unique(selected_author_organisation_edge_idx[1])
    selected_author_organisation_edge_idx, selected_org_indices = get_form_hetero_graph(hetero_data, (
        'author', 'affiliated_with', 'organization'), authors_idx)
    hetero_data['organization'].x = hetero_data['organization'].x[selected_org_indices]
    hetero_data.organization_text_features = [hetero_data.organization_text_features[i] for i in selected_org_indices]
    # TODO unify function
    paper_venue_edge_idx = hetero_data['paper', 'published_in', 'venue'].edge_index
    venue_mask = torch.isin(paper_venue_edge_idx[0], selected_paper_indices)
    selected_paper_venue_edge_idx = paper_venue_edge_idx[:, venue_mask]
    selected_venue_indices = torch.unique(selected_paper_venue_edge_idx[1])
    hetero_data['venue'].x = hetero_data['venue'].x[selected_venue_indices]
    hetero_data.venue_text_features = [hetero_data.venue_text_features[i] for i in selected_venue_indices]

    author_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(authors_idx.tolist())}
    paper_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_paper_indices.tolist())}
    venue_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_venue_indices.tolist())}
    organisation_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_org_indices.tolist())}

    new_author_paper_edge_index = selected_author_paper_edge_idx.clone()
    for i in range(selected_author_paper_edge_idx.shape[1]):
        new_author_paper_edge_index[0, i] = author_mapping[selected_author_paper_edge_idx[0, i].item()]
        new_author_paper_edge_index[1, i] = paper_mapping[selected_author_paper_edge_idx[1, i].item()]

    new_author_organisation_index = selected_author_organisation_edge_idx.clone()
    for i in range(selected_author_organisation_edge_idx.shape[1]):
        new_author_organisation_index[0, i] = author_mapping[selected_author_organisation_edge_idx[0, i].item()]
        new_author_organisation_index[1, i] = organisation_mapping[selected_author_organisation_edge_idx[1, i].item()]

    new_paper_venue_edge_index = selected_paper_venue_edge_idx.clone()
    for i in range(selected_paper_venue_edge_idx.shape[1]):
        new_paper_venue_edge_index[0, i] = paper_mapping[selected_paper_venue_edge_idx[0, i].item()]
        new_paper_venue_edge_index[1, i] = venue_mapping[selected_paper_venue_edge_idx[1, i].item()]

    hetero_data['author', 'affiliated_with', 'organization'].edge_index = new_author_organisation_index
    hetero_data['author', 'writes', 'paper'].edge_index = new_author_paper_edge_index
    hetero_data['paper', 'published_in', 'venue'].edge_index = new_paper_venue_edge_index

    return hetero_data


def check_features(hetero_data):
    array = ['author_text_features', 'venue_text_features', 'paper_text_features', 'organization_text_features']
    for text_features in array:
        if text_features in hetero_data:
            print(len(hetero_data[text_features]), f"is len of {text_features}")
        else:
            print(f"No {text_features}")


def main():
    args = parse_args()
    # Загрузка данных
    data_loader = DataLoader(args.root)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Построение графа
    graph_builder = GraphBuilder(data_loader)
    hetero_data = graph_builder.build_hetero_data()
    print(hetero_data)
    check_features(hetero_data)
    hetero_data.validate()

    data_loader.load_test_data()
    test_idx = data_loader.load_tset_idx()
    hetero_data = sub_graph_for_testing(hetero_data, torch.tensor(test_idx, dtype=torch.long))
    print(hetero_data)
    check_features(hetero_data)
    hetero_data.validate()

    # Создание дубликатов
    hetero_data_with_duplicates = graph_builder.create_duplicates(hetero_data, 2)
    print("Граф построен:")
    print(hetero_data_with_duplicates)
    check_features(hetero_data_with_duplicates)
    hetero_data_with_duplicates.validate()

    # author_embedding, paper_embedding, venue_embedding, organization_embedding \
    #     = text_to_bert_embedding(hetero_data_with_duplicates.author_text_features,
    #                              hetero_data_with_duplicates.paper_text_features,
    #                              hetero_data_with_duplicates.venue_text_features,
    #                              hetero_data_with_duplicates.organization_text_features,
    #                              root=args.output_dir, model_name=args.model_name, device=args.device)
    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer(args.model_name, cache_folder=os.path.join(args.output_dir, "sbert"), device=args.device)
    author_embedding = text_to_embeddings(hetero_data_with_duplicates.author_text_features, bert, 32)
    paper_embedding = text_to_embeddings(hetero_data_with_duplicates.paper_text_features, bert, 32)
    venue_embedding = text_to_embeddings(hetero_data_with_duplicates.venue_text_features, bert, 32)
    organization_embedding = text_to_embeddings(hetero_data_with_duplicates.organization_text_features, bert, 32)
    hetero_data_with_duplicates['author'].x = author_embedding
    hetero_data_with_duplicates['paper'].x = paper_embedding
    hetero_data_with_duplicates['venue'].x = venue_embedding
    hetero_data_with_duplicates['organization.x'].x = organization_embedding

    # Определение типов узлов и ребер
    node_types = ['author', 'paper', 'organization', 'venue']
    edge_types = [
        ('author', 'writes', 'paper'),
        ('paper', 'published_in', 'venue'),
        ('author', 'affiliated_with', 'organization'),
        ('author', 'duplicate', 'author')
    ]

    hetero_data_with_duplicates_and_false_edges = graph_builder.prepare_train_data(hetero_data_with_duplicates)
    print(hetero_data_with_duplicates_and_false_edges)
    check_features(hetero_data_with_duplicates_and_false_edges)
    hetero_data_with_duplicates_and_false_edges.validate()
    # Создание модели

    model = DuplicatePredictor(
        hidden_channels=args.hidden_channels,
        num_node_types=node_types,
        num_edge_types=edge_types,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Обучение модели
    trained_model, test_loss, test_acc = train_model(model, hetero_data_with_duplicates_and_false_edges,
                                                     epochs=args.epochs,
                                                     learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                                     device=args.device, save_root_path=args.output_dir)

    print(f'\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    print("Обучение завершено!")


if __name__ == "__main__":
    main()

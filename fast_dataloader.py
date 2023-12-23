import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp


def read_graph(path):
    G = nx.Graph()

    with open(path) as f:
        assert ":" in next(f)
        for line in f:
            if line == "\n":
                break
            u, v = list(map(int, line[:-1].split()))
            G.add_edge(u, v)

        assert ":" in next(f)
        for line in f:
            if line == "\n":
                break
            u, atom = list(map(int, line[:-1].split()))
            G.add_node(u, atom=atom)

    return G


import matplotlib.pyplot as plt


def show(G, path):
    fig = plt.figure()
    ax = fig.add_subplot()
    pos = nx.planar_layout(G)
    pos = nx.spring_layout(G, pos=pos)
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig(path)


def process_graph(G: nx.Graph):
    G_original = G.copy()
    A = nx.adjacency_matrix(G)
    A_sq = A @ A
    A_sq.setdiag(0)
    A_sq = A_sq != 0
    G2 = nx.from_scipy_sparse_array(A_sq)
    nx.set_node_attributes(G2, nx.get_node_attributes(G_original, "atom"), "atom")

    # cycles = [cycle for cycle in nx.cycle_basis(G) if len(cycle) <= 6]
    # # show(G, "before.png")

    # cycles_of_edge = defaultdict(list)
    # for i, cycle in enumerate(cycles):
    #     for j in range(len(cycle)):
    #         u, v = cycle[j], cycle[(j + 1) % len(cycle)]
    #         cycles_of_edge[(u, v)].append(i)
    #         cycles_of_edge[(v, u)].append(i)

    # for i, cycle in enumerate(cycles):
    #     G.add_node(-i - 1, atom=cycle)
    #     for u in cycle:
    #         for v in G[u]:
    #             if (u, v) in cycles_of_edge:
    #                 for other_cycle in cycles_of_edge[(u, v)]:
    #                     if other_cycle != i:
    #                         G.add_edge(-i - 1, -other_cycle - 1)
    #             else:
    #                 G.add_edge(-i - 1, v)

    # for cycle in cycles:
    #     for u in cycle:
    #         if u in G.nodes:
    #             G.remove_node(u)
    # show(G, "after0.png")
    G = nx.disjoint_union_all([G_original, G2])
    # G = nx.convert_node_labels_to_integers(G)
    # show(G, "after.png")
    # exit()
    return G


def graph_to_tensors(G, atom_embedding):
    G = process_graph(G)
    edge_index = list(G.to_directed().edges)
    edge_index = torch.LongTensor(edge_index).reshape(-1, 2).T

    atoms = [G.nodes[u]["atom"] for u in sorted(G.nodes)]
    x = []
    unk = atom_embedding["UNK"]
    # for atom in atoms:
    #     emb = np.zeros(6 * unk.shape[0])
    #     if type(atom) is list:
    #         atom = [atom_embedding.get(a, unk) for a in atom]
    #         atom = np.concatenate(atom)
    #         emb[: atom.shape[0]] = atom
    #         x.append(emb)
    #     else:
    #         emb[: unk.shape[0]] = atom_embedding.get(atom, unk)
    #         x.append(emb)
    for atom in atoms:
        x.append(atom_embedding.get(str(atom), unk))
    x = np.stack(x, axis=0)
    x = torch.FloatTensor(x).half()
    return x, edge_index


class GraphTextDataset(InMemoryDataset):
    def __init__(
        self, root, atom_embedding, split, tokenizer=None, transform=None, pre_transform=None
    ):
        self.root = root
        self.atom_embedding = atom_embedding
        self.split = split
        self.tokenizer = tokenizer
        description = pd.read_csv(os.path.join(self.root, f"{split}.tsv"), sep="\t", header=None)
        self.cids = description[0].to_list()
        self.description = description.set_index(0).to_dict()[1]

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{cid}.graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    def download(self):
        assert False

    def process(self):
        tot_atoms, tot_unk = 0, 0
        data_list = []
        for cid, path in tqdm(zip(self.cids, self.raw_paths), total=len(self.cids)):
            assert str(cid) in path

            text_input = self.tokenizer(
                [self.description[cid]],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length",
                add_special_tokens=True,
            )

            G = read_graph(path)
            x, edge_index = graph_to_tensors(G, self.atom_embedding)

            data = Data(
                x=x,
                edge_index=edge_index,
                input_ids=text_input["input_ids"],
                attention_mask=text_input["attention_mask"],
            )
            data_list.append(data)

        if tot_unk > 0:
            print(f"{tot_unk} atoms were unknown ({100*tot_unk/tot_atoms:.3f}%)")
        self.save(data_list, self.processed_paths[0])


class GraphDataset(InMemoryDataset):
    def __init__(
        self, root, atom_embedding, split, tokenizer=None, transform=None, pre_transform=None
    ):
        self.root = root
        self.atom_embedding = atom_embedding
        self.split = split
        self.tokenizer = tokenizer
        description = pd.read_csv(os.path.join(self.root, f"{split}.txt"), sep="\t", header=None)
        self.cids = description[0].to_list()

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{cid}.graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    def download(self):
        assert False

    def process(self):
        tot_atoms, tot_unk = 0, 0
        data_list = []
        for cid, path in tqdm(zip(self.cids, self.raw_paths), total=len(self.cids)):
            assert str(cid) in path

            G = read_graph(path)
            x, edge_index = graph_to_tensors(G, self.atom_embedding)

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        if tot_unk > 0:
            print(f"{tot_unk} atoms were unknown ({100*tot_unk/tot_atoms:.3f}%)")
        self.save(data_list, self.processed_paths[0])


class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

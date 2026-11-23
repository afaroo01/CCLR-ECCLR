import torch

from plantdisease.data import DataModule, DatasetSplitter, LeafImageDataset, TaskSampler


def test_class_map_merges_and_ignores_unmapped(load):
    cfg = load("experiments/supervised/exp1_efficientnet_autoaugment.yaml")
    ds = LeafImageDataset(cfg.datasets["plantvillage"].roots, cfg.data.classes,
                          cfg.datasets["plantvillage"].class_map)
    assert len(ds) == 8 * 20  # Corn folder ignored
    target = LeafImageDataset(cfg.datasets["plantdoc"].roots, cfg.data.classes,
                              cfg.datasets["plantdoc"].class_map)
    assert set(target.class_counts().values()) == {10}  # train + test merged


def test_splitter_is_deterministic_and_disjoint():
    labels = [i % 8 for i in range(100)]
    a = DatasetSplitter([0.7, 0.2, 0.1], 42).split(labels)
    b = DatasetSplitter([0.7, 0.2, 0.1], 42).split(labels)
    assert a == b and (len(a.train), len(a.val), len(a.test)) == (70, 20, 10)
    assert not set(a.train) & set(a.val) | set(a.train) & set(a.test)


def test_cutmix_loader_gives_soft_labels(load):
    cfg = load("experiments/supervised/exp2_efficientnet_cutmix.yaml")
    images, targets = next(iter(DataModule(cfg, 0).loader("train")))
    assert images.shape == (8, 3, 64, 64) and targets.shape == (8, 8)


def test_task_sampler_episode_shapes_and_class_ids():
    labels = [i % 4 for i in range(40)]
    sampler = TaskSampler(labels, n_way=3, n_shot=2, n_query=3, n_tasks=2, seed=0)
    task = next(iter(sampler))
    batch = [(torch.full((3, 4, 4), float(labels[i])), labels[i]) for i in task]
    s, sl, q, ql, ids = sampler.episodic_collate_fn(batch)
    assert s.shape[0] == 6 and q.shape[0] == 9 and len(ids) == 3
    # every query image's pixel value equals its global label -> mapping is consistent
    assert torch.equal(ids[ql].float(), q[:, 0, 0, 0])

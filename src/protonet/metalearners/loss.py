import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def get_prototypes(embeddings, targets):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    """

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return targets.eq(c).nonzero().squeeze(1)

    classes = torch.unique(targets)
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([embeddings[idx_list].mean(0) for idx_list in support_idxs])

    return prototypes


def prototypical_loss(prototypes, test_embeddings, targets, n_query):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    classes = torch.unique(targets)
    n_classes = len(classes)
    # query_idxs = torch.stack(
    #     list(map(lambda c: targets.eq(c).nonzero(), classes))).view(-1)
    #
    # query_samples = test_embeddings.to('cpu')[query_idxs]
    query_samples = test_embeddings.to('cpu')
    dists = euclidean_dist(query_samples, prototypes.to('cpu'))

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

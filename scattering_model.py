import torch

def forward_cross_section_model(T, c2, alpha, cstaralpha):
    T = T[:, None, :]
    cstaralpha_t= torch.sum(cstaralpha * T, dim=2)
    alpha_t = torch.sum(alpha * T, dim = 2)

    sigma = c2 + 2 * cstaralpha_t.real + torch.abs(alpha_t) ** 2

    return sigma
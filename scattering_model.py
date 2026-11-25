import torch

def forward_cross_section_model(T, C, alpha):
    """
    Assumptions:
    C: shape (N, A)
    alpha: shape (N, A, B)
    T: shape (N, B)
    """
    
    cross_section_amplitude = C + torch.einsum(
        "nab, nb -> na", alpha, T
    )

    cross_section = torch.sum(torch.abs(cross_section_amplitude)**2, dim=1)

    return cross_section


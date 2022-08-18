"""NCL utility functions"""
import torch

def additive_nearest_kf(B, C):
    """Here it is assumed that all these matrices are symmetric, which is NOT CHECKED explicitly"""
    BR, BL = B["A"], B["G"]
    CR, CL = C["A"], C["G"]

    trBL, trBR, trCL, trCR = (
        torch.trace(BL),
        torch.trace(BR),
        torch.trace(CL),
        torch.trace(CR),
    )
    if min(trBL, trBR) <= 0:
        print("zero trace!")
        return CR, CL
    elif min(trCL, trCR) <= 0:
        print("zero trace!")
        return BR, BL

    pi = torch.sqrt(torch.trace(BL) * torch.trace(CR)) / torch.sqrt(torch.trace(CL) * torch.trace(BR))

    return BR + CR / pi, BL + CL * pi

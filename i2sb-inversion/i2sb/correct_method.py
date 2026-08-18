import torch

from utils.utils import Emat_xyt, c2r, r2c

_CORRECT_METHOD = {}


def Get_Method(key):
    print("log %s_method" % key)
    return _CORRECT_METHOD[key]

def register_correct_method(func=None, name=None):
    def _register(func):
        if name is None:
            local_name = func.__name__
        else:
            local_name = name
        if local_name in _CORRECT_METHOD:
            raise ValueError(
                f"Already registered correct method with name: {local_name}"
            )
        _CORRECT_METHOD[local_name] = func
        return func

    if func is None:
        return _register
    else:
        return _register(func)

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """

    def __init__(self, csm, mask, lam, num=True):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(
            torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.0).float()
        )
        self.lam = lam
        self.num = num

    def myAtA(self, img):
        # print(img.size())
        x = Emat_xyt(img, False, self.csm, self.mask, num=self.num)
        x = Emat_xyt(x, True, self.csm, self.mask, num=self.num)
        # print(x.size())

        return x + self.lam * img

def myCG(A, Rhs, x0, it):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    # print('Rhs1', Rhs.shape, Rhs.dtype) #Rhs1.shape torch.Size([2, 256, 232])

    Rhs = r2c(Rhs) + A.lam * r2c(x0)

    # x = torch.zeros_like(Rhs)
    x = r2c(x0)
    i = 0
    r = Rhs - r2c(A.myAtA(x0))
    p = r
    rTr = torch.sum(torch.conj(r) * r).float()

    while i < it:
        Ap = r2c(A.myAtA(c2r(p)))
        alpha = rTr / torch.sum(torch.conj(p) * Ap).float()
        alpha = torch.complex(alpha, torch.tensor(0.0).float().cuda())
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r) * r).float()
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.0).float().cuda())
        p = r + beta * p
        i = i + 1
        rTr = rTrNew

    return c2r(x)

def normals(x):
    temp = r2c(x)
    minv = torch.std(temp)
    temp = temp / (1.5 * minv)
    return temp, minv

def back_normal(x, minv):
    return x * 1.5 * minv

@register_correct_method(name="CG")
# [x1=T1_img,mask,csm=t2_csm,yk=欠采的t2_img]
def CG_Method(pred_x0, mask, yk, x1, csm,cg_weight,cg_iter):

    # pred_x0, minv_p0 = normals(pred_x0)
    # minv_p0 = r2c(pred_x0).abs().max()
    # pred_x0 = r2c(pred_x0)/minv_p0
    pred_x0 = r2c(pred_x0)
    temp = r2c(yk)

    # temp, minv_x1 = normals(yk if "yk" in locals() else x1)
    # 参数待调整
    # CGWEIGHT = next(label_list) 0.5
    # CGWEIGHT =0.00005   # inversion 0.07 ddpm0.05 0.5 0.001
    CGWEIGHT = cg_weight
    # 0.00005  0.0002219828893430531
    Aobj = Aclass(
        c2r(csm),
        mask.to("cuda"),
        torch.tensor(CGWEIGHT).cuda(),
        num=True,
    )
    Rhs = c2r(temp)
    pred_x0 = pred_x0
    # print(f"Starting CG with max_iter=5")
    pred_x0 = myCG(Aobj, Rhs, c2r(pred_x0),cg_iter)
    # print(f"CG completed")
    # pred_x0 = back_normal(r2c(pred_x0), minv_p0)
    # pred_x0 = r2c(pred_x0)*minv_p0
    pred_x0 = r2c(pred_x0)
    pred_x0 = c2r(pred_x0).type(torch.FloatTensor).to("cuda")
    return pred_x0

@register_correct_method(name="CSM")
def CSM_method(pred_x0, mask, yk, x1, csm):
    pred_x0, minv_p0 = normals(pred_x0)
    temp, minv_x1 = normals(yk if "yk" in locals() else x1)
    
    # pred_x0 = r2c(pred_x0)
    # temp = r2c(yk)
    # pred_x0 = (1 - mask.to(pred_x0)) * fft2c_2d(((pred_x0))) + fft2c_2d(
    #     ((temp))
    # )
    # pred_x0 = ifft2c_2d(pred_x0)

    pred_x0 = (1 - mask.to(pred_x0)) * fft2c_2d(
        ((pred_x0)) * (csm).to(pred_x0)
    ) + fft2c_2d(((temp)) * (csm).to(pred_x0))
    pred_x0 = Emat_xyt_complex(pred_x0, True, csm.to(pred_x0), 1)
    pred_x0 = back_normal(pred_x0, minv_p0)
    pred_x0 = c2r(pred_x0)
    return pred_x0

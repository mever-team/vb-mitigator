from .acc import acc_dict, acc
from .unb_bc_ba import unb_bc_ba_dict, unb_bc_ba
from .wg_ovr import wg_ovr_dict, wg_ovr


metrics_dicts = {
    "acc": acc_dict,
    "unb_bc_ba": unb_bc_ba_dict,
    "wg_ovr": wg_ovr_dict,
}

get_performance = {
    "acc": acc,
    "unb_bc_ba": unb_bc_ba,
    "wg_ovr": wg_ovr,
}
